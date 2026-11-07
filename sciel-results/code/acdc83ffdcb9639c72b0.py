"""Symbolic regression: fit an orthogonal polynomial basis (Legendre) via BFGS."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_BUDGET = 135.0  # safety margin before 150s kill


def _remaining_time() -> float:
    """Return remaining seconds before safety deadline."""
    return _TIME_BUDGET - (time.time() - _START_TIME)


def _legendre_basis(x):
    """
    Compute Legendre polynomial basis functions P_0(t) ... P_9(t)
    where t = x / 2.0 maps x from [-2, 2] to [-1, 1].
    
    Legendre polynomials are orthogonal with respect to the uniform measure
    on [-1, 1], which matches the uniform distribution of the input data.
    This results in a nearly diagonal Gram matrix, ensuring excellent
    numerical conditioning for the least-squares fit.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    num_terms = 10
    
    # Map to [-1, 1]
    t = x / 2.0
    
    # Initialize basis matrix
    basis = np.zeros((n, num_terms), dtype=np.float64)
    
    if num_terms > 0:
        basis[:, 0] = 1.0
    if num_terms > 1:
        basis[:, 1] = t
        
    # Recurrence relation:
    # (k + 1) P_{k+1}(t) = (2k + 1) t P_k(t) - k P_{k-1}(t)
    for k in range(1, num_terms - 1):
        basis[:, k + 1] = ((2.0 * k + 1.0) * t * basis[:, k] - k * basis[:, k - 1]) / (k + 1.0)
        
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluate the model: sum of params[i] * P_i(x/2).
    This is linear in params. The orthogonal basis ensures that BFGS
    will converge rapidly to the global minimum of the MSE.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    # Ensure params has exactly 10 elements
    if len(params) < 10:
        params = np.concatenate([params, np.zeros(10 - len(params))])
    else:
        params = params[:10]
        
    basis = _legendre_basis(x)
    return basis @ params
# EVOLVE-BLOCK-END
