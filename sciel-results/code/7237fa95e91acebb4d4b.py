import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Orthonormalized Chebyshev polynomial basis for symbolic regression.
    
    params: array of length 10, coefficients for the orthonormalized Chebyshev basis.
    x: array of input points in [-3, 3].
    
    We map x from [-3, 3] to [-1, 1] and evaluate orthonormalized Chebyshev
    polynomials. The orthonormalization ensures well-conditioning for BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    t = x / 3.0  # Map to [-1, 1]
    
    n = 10
    # Evaluate Chebyshev polynomials T_0 through T_9 on t
    T = np.zeros((len(t), n), dtype=np.float64)
    T[:, 0] = 1.0
    if n > 1:
        T[:, 1] = t
    for k in range(2, n):
        T[:, k] = 2.0 * t * T[:, k-1] - T[:, k-2]
    
    # Orthonormalize columns using QR decomposition
    # This ensures the basis is well-conditioned for BFGS
    Q, R = np.linalg.qr(T)
    
    # Return Q @ params
    return Q @ params
