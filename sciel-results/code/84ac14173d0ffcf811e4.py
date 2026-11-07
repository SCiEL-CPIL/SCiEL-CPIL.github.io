"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: track start time at module level
_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill


def _elapsed_ok():
    """Check if we're still within deadline."""
    return (time.time() - _START_TIME) < _DEADLINE


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y using a degree-9 Chebyshev polynomial fit.
    x: input array
    params: array of 10 coefficients
    Returns: predictions array same shape as x
    """
    if not _elapsed_ok():
        # Return a safe fallback if over deadline
        return np.zeros_like(x)
    
    # Construct a diverse basis of functions to allow fitting various target shapes
    # The parameters are linear weights for these basis functions.
    # Basis: 1, x, x^2, x^3, x^4, sin(x), cos(x), exp(x), tanh(x), 1/(1+x^2)
    
    # Ensure x is float for operations
    x = np.asarray(x, dtype=float)
    
    # Calculate basis functions
    basis = np.column_stack((
        np.ones_like(x),
        x,
        x**2,
        x**3,
        x**4,
        np.sin(x),
        np.cos(x),
        np.exp(x),
        np.tanh(x),
        1.0 / (1.0 + x**2)
    ))
    
    # Linear combination: y = basis @ params
    # params has shape (10,), basis has shape (N, 10)
    y = basis @ params
    
    return y
# EVOLVE-BLOCK-END
