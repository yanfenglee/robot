## 线性高斯模型(Linear-Gaussian model, LG)

运动模型：$x_k = A_{k-1}x_{k-1} + v_k + w_k$  
观测模型：$y_k = C_kx_k + n_k$  
  
其中各符号含义如下：  
  
离散时间索引: $k \in \mathbb{N}$  
系统状态: $x_k \in \mathbb{R}^N$  
输入: $v_k \in \mathbb{R}^N$  
处理噪声: $w_k \in \mathbb{R}^N \sim (0, Q_k)$  
观测噪声: $n_k \in \mathbb{R}^N \sim (0, R_k)$  
转换矩阵: $A_k \in \mathbb{R}^{N*N}$  
观测矩阵: $C_k \in \mathbb{R}^{M*N}$  

状态估计问题定义为：根据系统初始值，输入量，观测量，按上面的模型寻找X的最佳的估计值  
  
## 求解方法
* 贝叶斯推断(Bayesian inference)
* 最大化后验(MAP)
  
对于LG问题，上面两种方法是一样的，完全贝叶斯后验就是高斯后验，X在均值处取得最大概率  

### MAP
$$
\begin{aligned}  
\hat{x} &= argmax_xp(x|v,y) \\  
        &= argmax_x\frac{p(y|v,x)*p(x|v)}{p(y|v)} \\  
        &= argmax_xp(y|x)*p(x|v)  
\end{aligned}  
$$
其中输入v包括系统的初始状态和输入序列,y为观测值序列, 可以丢掉分母，因为p(y|v)不依赖x，p(y|v,x)丢掉v因为y不依赖v  
  
由于w,n噪声的不相关性, 可对p(y|x)，p(x|v) 进行分解：  
$$
p(y|x)=\prod_{k=1}^Kp(y_k|x_k) \\  
p(x|v)=\prod_{k=1}^Kp(x_k|x_{k-1},v_k)  
$$

根据高斯分布有：  
$$\begin{aligned}  
p(y_k|x_k)&=\frac{1}{\sqrt{(2\pi)^NdetR_k}}exp(-\frac{1}{2}(y_k-C_kx_k)^TR_k^{-1}(y_k-C_kx_k)) \\  
p(x_k|x_{k-1},v_k)&=\frac{1}{\sqrt{(2\pi)^NdetQ_k}}exp(-\frac{1}{2}(x_k-A_{k-1}x_{k-1}-v_k)^TQ_k^{-1}(x_k-A_{k-1}x_{k-1}-v_k))
\end{aligned}$$
  
方便起见对p(y|x)，p(x|v)取对数，并去掉不依赖x的量, 去掉负号：  
$$\begin{aligned}
J_{v,k}(x)&=\frac{1}{2}(x_k-A_{k-1}x_{k-1}-v_k)^TQ_k^{-1}(x_k-A_{k-1}x_{k-1}-v_k) \\  
J_{y,k}(x)&=\frac{1}{2}(y_k-C_kx_k)^TR_k^{-1}(y_k-C_kx_k)
\end{aligned}$$
  
目标函数可以重新定义为：  
$$
J(x)=\sum_{k=0}^{K}(J_{v,k}(x)+J_{y,k}(x))  
$$
  
把数据写成向量或矩阵形式，令：  
$$
Z=\left[
    \begin{array}{ccc}
    v_1\\  
    v_2\\  
    ...\\  
    v_K\\  
    y_1\\  
    y_2\\  
    ...\\  
    y_K
\end{array}
\right],

H=\left[\begin{array}{ccc}
1\\  
-A_1    &...\\  
        &A_{K-1}    &1\\  
        \\  
C_1\\  
        &...\\  
        &      &C_K\\  
\end{array}\right],


X=\left[\begin{array}{ccc}
    x_1\\  
    x_2\\  
    ...\\  
    x_K  
\end{array}\right],

W=\left[\begin{array}{ccc}
    Q_1\\  
        &...\\
        &   &Q_K\\
        &   &   &R_1\\
        &   &   &   &...\\
        &   &   &   &   &R_K\\
\end{array}\right]
  
$$

估计问题转化为如下的无约束优化问题：  
$$
J(X)=\frac{1}{2}(Z-HX)^TW^{-1}(Z-HX)  
$$
$$
\hat{X}=argmin_XJ(X)  
$$