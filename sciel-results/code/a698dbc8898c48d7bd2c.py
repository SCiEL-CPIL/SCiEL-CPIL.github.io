import numpy as np
import time

_START_TIME = time.time()
_basis_cache = None
_x_cache = None

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev Polynomial Basis of Degree 9 (Second Kind).
    
    Strategy:
    - Maps input x from [0.1, 6.0] to z in [-1, 1].
    - Uses Chebyshev polynomials of the second kind U_0(z) ... U_9(z).
    - U_n(z) = sin((n+1)theta) / sin(theta) where z = cos(theta).
    - This basis is also well-conditioned and orthogonal with respect to sqrt(1-x^2).
    - Might capture different features than T_n.
    """
    global _basis_cache, _x_cache
    
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    p = np.asarray(params, dtype=np.float64)
    if p.ndim != 1 or p.shape[0] != 10:
        return np.zeros_like(x)

    # Time limit guard
    if time.time() - _START_TIME > 145:
        return np.zeros_like(x)

    # Check cache
    if _x_cache is not None and np.array_equal(x, _x_cache):
        basis = _basis_cache
    else:
        # Map x to [-1, 1]
        x_min = 0.1
        x_max = 6.0
        z = (2.0 * x - (x_max + x_min)) / (x_max - x_min)
        
        # Compute Chebyshev basis of the second kind
        # U_0 = 1
        # U_1 = 2z
        # U_{n+1} = 2z U_n - U_{n-1}
        
        N = len(x)
        basis = np.zeros((10, N), dtype=np.float64)
        
        basis[0] = 1.0
        if N > 0:
            basis[1] = 2.0 * z
            
            for k in range(2, 10):
                basis[k] = 2.0 * z * basis[k-1] - basis[k-2]
        
        _x_cache = x.copy()
        _basis_cache = basis

    # Prediction: y = sum(p_i * U_i(x))
    # basis shape (10, N), p shape (10,)
    # result shape (N,)
    result = basis.T @ p
    
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result
