"""Symbolic regression using Chebyshev basis with analytic least-squares solution."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_chebyt


def _cheb_basis(x: np.ndarray) -> np.ndarray:
    """Build Chebyshev basis matrix of degree 0..9 on x in [-2,2].
    Maps x from [-2,2] to t in [-1,1] for standard Chebyshev polynomials."""
    t = 0.5 * x  # map [-2,2] -> [-1,1]
    n = len(t)
    basis = np.empty((n, 10), dtype=np.float64)
    for k in range(10):
        basis[:, k] = eval_chebyt(k, t)
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: y = sum_{k=0}^{9} p_k * T_k(x/2)
    where T_k is the k-th Chebyshev polynomial of the first kind.

    Since the model is linear in parameters, BFGS will converge to the
    least-squares optimum. We provide a well-conditioned Chebyshev basis
    on [-2,2] mapped to [-1,1].

    params (length 10): coefficients for Chebyshev polynomials T_0 through T_9.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    # Build Chebyshev basis
    t = 0.5 * x  # map [-2,2] -> [-1,1]
    n = len(t)

    # Compute Chebyshev polynomials T_0..T_9 via recurrence
    # T_0(t) = 1, T_1(t) = t, T_{k}(t) = 2t*T_{k-1}(t) - T_{k-2}(t)
    T0 = np.ones(n, dtype=np.float64)
    T1 = t.copy()
    result = params[0] * T0 + params[1] * T1

    if n > 0:
        for k in range(2, 10):
            Tk = 2.0 * t * T1 - T0
            result = result + params[k] * Tk
            T0 = T1
            T1 = Tk

    # Ensure finite output
    return np.nan_to_num(result, nan=0.0, posinf=1e8, neginf=-1e8)
# EVOLVE-BLOCK-END
