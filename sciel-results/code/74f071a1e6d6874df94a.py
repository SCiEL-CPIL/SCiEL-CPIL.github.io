"""Symbolic regression with 1 input using Chebyshev polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Fit a linear combination of Chebyshev polynomials of the first kind (degrees 0-9).
    Chebyshev polynomials provide a well-conditioned basis on a bounded interval,
    which is ideal for BFGS optimization. They span the same space as a degree-9
    polynomial but with much better numerical properties.
    
    The input x is mapped from [0.1, 6.0] to [-1, 1] where Chebyshev polynomials
    are defined.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if len(x) == 0:
        return np.empty(0, dtype=np.float64)
    
    # Map x from [0.1, 6.0] to u in [-1, 1]
    x_min = 0.1
    x_max = 6.0
    center = (x_min + x_max) / 2.0
    scale = (x_max - x_min) / 2.0
    
    # Clip to valid domain to avoid issues at boundaries
    x_safe = np.clip(x, x_min, x_max)
    u = (x_safe - center) / scale
    
    # Compute Chebyshev polynomials T_0 to T_9
    # T_0(u) = 1
    # T_1(u) = u
    # T_k(u) = 2*u*T_{k-1}(u) - T_{k-2}(u)
    
    basis = np.empty((len(u), 10), dtype=np.float64)
    
    basis[:, 0] = 1.0
    basis[:, 1] = u
    
    for k in range(2, 10):
        basis[:, k] = 2.0 * u * basis[:, k-1] - basis[:, k-2]
    
    # Linear combination of basis functions with parameters
    return basis @ params
# EVOLVE-BLOCK-END
