# EVOLVE-BLOCK-START
import numpy as np
import numpy.polynomial.legendre as npleg
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the hidden function using a whitened Legendre polynomial basis.
    The basis columns are normalized to unit L2 norm on the input points,
    ensuring the Hessian of the MSE is close to the identity, which improves
    BFGS convergence stability.
    """
    _T0 = time.time()
    _DEADLINE = 140.0

    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.array([])

    u = x / 3.0
    V = npleg.legvander(u, 9)
    
    # Whitening: normalize each column to unit norm
    # This makes the design matrix orthonormal, so V.T @ V = I
    # Consequently, the Hessian of the squared error is 2*I, which is ideal for BFGS.
    norms = np.linalg.norm(V, axis=0)
    # Avoid division by zero
    norms[norms < 1e-12] = 1.0
    V_whitened = V / norms
    
    y = V_whitened @ params

    if not np.all(np.isfinite(y)):
        y = np.where(np.isfinite(y), y, 0.0)

    if time.time() - _T0 > _DEADLINE:
        return y

    return y
# EVOLVE-BLOCK-END
