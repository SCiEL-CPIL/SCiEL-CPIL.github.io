"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Uses a shifted Legendre polynomial basis (degree 0-9) on [0.1, 6].
    The basis is orthonormal, making the linear-in-params model well-conditioned.
    BFGS starting from all-ones should converge to the global minimum efficiently.
    """
    # Map x from [0.1, 6.0] to [0, 1] for shifted Legendre
    x_min = 0.1
    x_max = 6.0
    t = (x - x_min) / (x_max - x_min)
    t = np.clip(t, 0.0, 1.0)
    
    # Compute shifted Legendre polynomials P_0 to P_9
    # P_0(t) = 1
    # P_1(t) = 2t - 1
    # P_n(t) = (2n-1)(2t-1)P_{n-1}(t) - (n-1)P_{n-2}(t)
    # But shifted Legendre: P_n(2t-1) where P_n are standard Legendre
    # Recurrence for shifted: P_0=1, P_1=2t-1
    # P_n = ((2n-1)(2t-1)P_{n-1} - (n-1)P_{n-2}) / n
    
    n_points = x.shape[0]
    basis = np.empty((n_points, 10), dtype=np.float64)
    basis[:, 0] = 1.0
    basis[:, 1] = 2.0 * t - 1.0
    
    for k in range(2, 10):
        # Shifted Legendre recurrence:
        # P_k(t) = ((2k-1)(2t-1)P_{k-1}(t) - (k-1)P_{k-2}(t)) / k
        basis[:, k] = ((2.0 * k - 1.0) * (2.0 * t - 1.0) * basis[:, k-1] - (k - 1.0) * basis[:, k-2]) / k
        
    return basis @ params
# EVOLVE-BLOCK-END
