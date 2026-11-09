import numpy as np
import time
import scipy.optimize as opt

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Multi-Start Gaussian RBF Basis Selection.
    
    The standard BFGS optimizer in the evaluator starts at params=[1,1,...,1].
    For non-linear basis functions (like Gaussians or Hinges), this start point
    is often poor, leading to local minima. 
    
    This program uses the 10 'params' provided by the evaluator as a "seed" 
    for a multi-start local search, but primarily relies on a fixed set of 
    robust initial configurations for the basis function centers and widths.
    
    We use a Gaussian RBF basis:
    y(x) = w_0 + sum_{i=1}^{K} w_i * exp(-((x - c_i)^2) / (2 * s_i^2))
    
    To fit within the 10-parameter constraint imposed by the evaluator's 
    minimize call, we cannot have free parameters for all centers/widths 
    optimized by BFGS simultaneously in a robust way from a bad start.
    
    Instead, we interpret the 'params' array as follows:
    The evaluator calls minimize on params. We will IGNORE the specific values 
    of params for the basis centers/widths during the initial search phase, 
    but we must return a function that is differentiable with respect to params 
    if we were to let BFGS refine them. However, the task says "equation(x, params)".
    
    Actually, a safer structural change that respects the interface:
    We use the 10 params to define a **Tunable Fourier Series** or **Polynomial**.
    Why? Because BFGS is very good at optimizing linear combinations of fixed basis functions.
    The previous attempt failed likely because the basis was non-convex (hinges) 
    and the start point was bad.
    
    Let's try a **Fixed Chebyshev Polynomial Basis of Degree 9**.
    Chebyshev polynomials are orthogonal on [-1, 1] (scaled to [-2, 2] here).
    They are numerically stable and form a linear basis.
    If the basis is fixed and linear in parameters, BFGS (or even gradient descent) 
    will converge to the global minimum of the MSE for that specific basis class.
    
    The previous "Inspiration 1" (score 190) was likely a bad polynomial.
    The "Inspiration 2" (Gaussian) scored 2.66.
    The "Inspiration 3" (Adaptive Gaussian) scored 1.28.
    
    Wait, the current program (Hinge) scored 7.09.
    The Adaptive Gaussian (score 1.28) is better. Why?
    Because it allowed the centers/widths to move.
    
    However, BFGS from all-ones is still risky for adaptive Gaussians.
    
    Let's try a **Hybrid Fixed Basis + Multi-Start Internal Optimization**.
    But `equation` is called *during* the optimization loop by BFGS.
    We cannot run a full multi-start search *inside* `equation` because it would be too slow 
    (150s limit, BFGS calls equation ~100-1000 times).
    
    Therefore, the strategy must be:
    1. Choose a basis that is **linear in the parameters** so BFGS converges reliably.
    2. Ensure the basis is **rich enough** to approximate the target.
    3. Map the domain [-2, 2] to [-1, 1] for stability.
    
    Let's use **Chebyshev Polynomials of the First Kind**, T_0 to T_9.
    This is a 10-dimensional linear basis.
    y = sum_{k=0}^{9} p_k * T_k((x+2)/2 - 1)
    
    This is structurally different from the Hinge basis (which is non-linear in params 
    if centers are params, but here centers are fixed? No, in the current program, 
    centers ARE params. That makes it non-linear and non-convex).
    
    By fixing the centers/locations of the basis functions and only letting BFGS 
    optimize the weights, we guarantee convergence to the best fit in that subspace.
    
    To make it "materially different" and potentially better than the current Hinge:
    The Hinge basis has 4 "kinks". The Chebyshev basis has 10 degrees of freedom 
    spread globally. If the target is smooth, Chebyshev will likely outperform Hinges 
    which struggle to fit smooth curves without many kinks. If the target has discontinuities, 
    Chebyshev will have Gibbs phenomena, but Hinges might be better.
    
    Given "noiseless" and "uniformly drawn", smooth functions are common in these benchmarks.
    Let's bet on smoothness and use a high-degree polynomial basis, specifically 
    **Lagrange Interpolation at Fixed Nodes** or **Chebyshev**.
    
    Let's go with **Chebyshev T_0..T_9**.
    """
    
    # Map x from [-2, 2] to [-1, 1] for Chebyshev stability
    # t = (x + 2) / 2 - 1 = x/2
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{n+1}(t) = 2t*T_n(t) - T_{n-1}(t)
    
    T = np.zeros((10, len(x)))
    T[0, :] = 1.0
    if len(x) > 0:
        T[1, :] = t
        for n in range(2, 10):
            T[n, :] = 2.0 * t * T[n-1, :] - T[n-2, :]
    
    # Linear combination
    # params is length 10
    # y = sum(p_k * T_k)
    result = T.dot(params)
    
    return result
