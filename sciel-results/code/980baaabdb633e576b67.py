"""Symbolic regression: fit a rich Chebyshev polynomial basis via BFGS."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_BUDGET = 135.0  # safety margin before 150s kill


def _remaining_time() -> float:
    """Return remaining seconds before safety deadline."""
    return _TIME_BUDGET - (time.time() - _START_TIME)


def _chebyshev_basis(x, n_terms=10):
    """
    Compute Chebyshev polynomials T_0(x) through T_{n_terms-1}(x).
    Input x is expected in [-2, 2]. We map it to [-1, 1] for standard Chebyshev.
    Returns a matrix of shape (len(x), n_terms).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    # Map x from [-2, 2] to [-1, 1]
    t = x / 2.0
    
    T = np.zeros((len(t), n_terms))
    T[:, 0] = 1.0
    if n_terms > 1:
        T[:, 1] = t
    for k in range(2, n_terms):
        T[:, k] = 2.0 * t * T[:, k - 1] - T[:, k - 2]
    return T


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluate the model: sum of params[i] * T_i(x/2).
    This is linear in params, so BFGS should converge well.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure params has exactly 10 elements
    if len(params) < 10:
        # Pad with zeros if needed
        params = np.concatenate([params, np.zeros(10 - len(params))])
    else:
        params = params[:10]
        
    # Compute the basis matrix
    basis = _chebyshev_basis(x, 10)
    
    # Compute predictions
    return basis @ params
# EVOLVE-BLOCK-END
