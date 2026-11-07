"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate using Chebyshev Polynomials.
    
    Strategy:
    The previous approach used localized Gaussian RBFs, which resulted in an 
    ill-conditioned basis matrix. This makes the optimization landscape for 
    BFGS difficult to navigate, often leading to poor convergence.
    
    This approach uses Chebyshev polynomials of the first kind, T_k(t), 
    defined on t in [-1, 1]. We map the input x in [-3, 3] to t in [-1, 1].
    
    Advantages:
    1. Chebyshev polynomials are orthogonal with respect to the weight 
       1/sqrt(1-t^2) on [-1, 1]. This orthogonality ensures that the 
       basis matrix is well-conditioned (condition number is bounded and 
       small compared to monomials or RBFs).
    2. A linear combination of Chebyshev polynomials up to degree N can 
       approximate any continuous function on [-1, 1] very efficiently.
    3. With 10 parameters, we can represent polynomials up to degree 9, 
       which is sufficient to capture many common smooth functions 
       (linear, quadratic, cubic, etc.) exactly or with high precision.
    4. BFGS can minimize the MSE of a linear model with a well-conditioned 
       basis very quickly, often reaching the global minimum in few iterations.
    
    Mapping:
    x in [-3, 3] -> t = x / 3.0 in [-1, 1].
    
    Basis:
    T_0(t) = 1
    T_1(t) = t
    T_2(t) = 2t^2 - 1
    ...
    T_n(t) = 2t T_{n-1}(t) - T_{n-2}(t)
    
    We compute the basis matrix B of shape (N, 10) where B[i, k] = T_k(x[i]/3).
    Prediction: pred = B @ params.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Ensure p has length 10
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)), mode='constant')
    else:
        p = p[:10]
        
    # Map x from [-3, 3] to [-1, 1]
    t = x / 3.0
    
    # Clip t to [-1, 1] to avoid numerical issues if x slightly exceeds range
    # though problem states uniform [-3, 3]
    t = np.clip(t, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0 = 1
    # T_1 = t
    # T_k = 2*t*T_{k-1} - T_{k-2}
    
    basis = np.empty((len(x), 10), dtype=np.float64)
    
    basis[:, 0] = 1.0
    if len(x) > 0:
        basis[:, 1] = t
        
    for k in range(2, 10):
        basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
        
    # Prediction is linear combination
    pred = basis @ p
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
        
    return pred
# EVOLVE-BLOCK-END
