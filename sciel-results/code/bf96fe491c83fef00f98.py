# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Returns prediction using a Legendre polynomial basis of degree 9.
    Legendre polynomials are orthogonal with respect to the uniform weight
    function, which matches the uniform distribution of inputs in [-3, 3].
    This provides a well-conditioned linear least-squares problem for BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Scale input to [-1, 1] for polynomial stability
    t = x / 3.0
    
    n = x.shape[0]
    basis = np.empty((n, 10), dtype=np.float64)
    
    # P_0(t) = 1
    basis[:, 0] = 1.0
    
    # P_1(t) = t
    basis[:, 1] = t
    
    # Legendre recurrence: (k+1)*P_{k+1}(t) = (2k+1)*t*P_k(t) - k*P_{k-1}(t)
    # So P_{k+1}(t) = [(2k+1)*t*P_k(t) - k*P_{k-1}(t)] / (k+1)
    for k in range(1, 9):
        basis[:, k+1] = ((2.0 * k + 1.0) * t * basis[:, k] - k * basis[:, k-1]) / (k + 1.0)
    
    return basis @ params
# EVOLVE-BLOCK-END
