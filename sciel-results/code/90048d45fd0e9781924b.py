"""Symbolic regression: fit a rich linear-in-parameters basis via BFGS."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_BUDGET = 135.0  # safety margin before 150s kill


def _remaining_time() -> float:
    """Return remaining seconds before safety deadline."""
    return _TIME_BUDGET - (time.time() - _START_TIME)


def _chebyshev(x, n):
    """Compute Chebyshev polynomials T_0(x) through T_{n-1}(x) mapped to [-1,1]."""
    # Map x from [-2, 2] to [-1, 1]
    t = x / 2.0
    T = np.zeros((len(t), n))
    T[:, 0] = 1.0
    if n > 1:
        T[:, 1] = t
    for k in range(2, n):
        T[:, k] = 2.0 * t * T[:, k - 1] - T[:, k - 2]
    return T


def _basis_functions(x):
    """
    Compute a diverse set of basis functions at points x.
    Returns a matrix of shape (len(x), num_basis).
    We use 10 basis functions to match the 10 params.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    
    # Chebyshev polynomials (well-conditioned) - 5 of them
    T = _chebyshev(x, 5)
    
    # Additional diverse functions
    # Sigmoid-like
    sigmoid = 1.0 / (1.0 + np.exp(-2.0 * x))
    # Rational bump (Lorentzian-like)
    bump = 1.0 / (1.0 + x * x)
    # Gaussian-like bump
    gauss = np.exp(-x * x)
    # Absolute value
    abs_x = np.abs(x)
    # Sign-like (tanh)
    tanh_x = np.tanh(x)
    
    # Stack all 10 basis functions
    basis = np.column_stack([
        T[:, 0],  # T0 = 1
        T[:, 1],  # T1
        T[:, 2],  # T2
        T[:, 3],  # T3
        T[:, 4],  # T4
        sigmoid,
        bump,
        gauss,
        abs_x,
        tanh_x,
    ])
    
    return basis


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Evaluate the model: sum of params[i] * basis_i(x).
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
    
    basis = _basis_functions(x)
    return basis @ params
# EVOLVE-BLOCK-END
