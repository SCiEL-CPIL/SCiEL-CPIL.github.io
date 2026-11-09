import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import eval_chebyt

# Deadline guard: kill at 145s to leave safety margin
_DEADLINE = 145.0
_t_start = time.time()

def _cheb_basis(t, max_degree=9):
    """Compute Chebyshev polynomials T_0..T_max_degree for array t in [-1,1]."""
    n = t.shape[0]
    T = np.zeros((n, max_degree + 1))
    T[:, 0] = 1.0
    if max_degree >= 1:
        T[:, 1] = t
    for k in range(2, max_degree + 1):
        T[:, k] = 2.0 * t * T[:, k - 1] - T[:, k - 2]
    return T

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression using a rich basis that BFGS can fit.
    We use Chebyshev polynomials up to degree 9 on t = x/2.
    """
    t = x / 2.0
    T = _cheb_basis(t, 9)
    pred = T @ params
    return pred
