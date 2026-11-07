import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Chebyshev Polynomial Approximation
    
    This approach replaces the non-linear RBF model with a linear basis of Chebyshev polynomials.
    
    Why this is different and potentially better:
    1. Linearity: The model is linear in parameters. BFGS is guaranteed to find the global
       minimum (the least-squares solution) without getting stuck in local minima, which is
       a major risk with learnable RBF widths/centers.
    2. Conditioning: Chebyshev polynomials are orthogonal on [-1, 1] and well-conditioned,
       unlike monomials (x^0...x^9) which become ill-conditioned for higher degrees.
    3. Expressiveness: A 9th-degree polynomial can approximate any continuous function on
       a closed interval reasonably well (Weierstrass theorem), capturing both global trends
       and some local curvature.
    4. Robustness: Since the fit is a simple linear regression in the transformed space,
       the "score 31k" failure mode of the previous non-linear model is avoided.
    
    Parameters (10 total):
    - c[0:10]: Coefficients for Chebyshev polynomials T_0(x') to T_9(x')
      where x' = (2x - (min+max)) / (max-min) maps [0.1, 6] to [-1, 1].
    """
    
    start_time = time.time()
    TIME_LIMIT = 140.0 # Safety margin

    x = np.asarray(x, dtype=np.float64).flatten()
    p = np.asarray(params, dtype=np.float64).flatten()
    
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
    else:
        p = p[:10]

    # Map x from [0.1, 6.0] to [-1, 1] for Chebyshev stability
    x_min, x_max = 0.1, 6.0
    x_norm = (2.0 * x - (x_min + x_max)) / (x_max - x_min)
    
    # Clip to [-1, 1] to handle any slight numerical overflow if x is slightly out of bounds
    x_norm = np.clip(x_norm, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(x) = 1
    # T_1(x) = x
    # T_{n}(x) = 2x T_{n-1}(x) - T_{n-2}(x)
    
    # Initialize basis matrix (N, 10)
    N = len(x_norm)
    basis = np.zeros((N, 10), dtype=np.float64)
    
    if N == 0:
        return np.array([])
        
    basis[:, 0] = 1.0
    basis[:, 1] = x_norm
    
    for k in range(2, 10):
        basis[:, k] = 2.0 * x_norm * basis[:, k-1] - basis[:, k-2]
        
    # Check time
    if time.time() - start_time > TIME_LIMIT:
        # Fallback: simple linear fit using first two params
        return p[0] + p[1] * x

    # Prediction: basis @ coeffs
    y = basis @ p
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        # Fallback to zero if something went wrong
        return np.zeros_like(x)
        
    return y
