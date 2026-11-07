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
    Predicts y using a linear combination of diverse basis functions.
    x: input array
    params: array of 10 coefficients
    Returns: predictions array same shape as x
    """
    if not _elapsed_ok():
        # Return a safe fallback if over deadline
        return np.zeros_like(x)
    
    # Ensure x is float for operations
    x = np.asarray(x, dtype=float)
    
    # Construct a diverse basis of functions to allow fitting various target shapes.
    # The parameters are linear weights for these basis functions.
    # Basis:
    # 0: 1 (constant)
    # 1: x (linear)
    # 2: x^2 (quadratic)
    # 3: x^3 (cubic)
    # 4: x^4 (quartic)
    # 5: sin(x) (low freq oscillation)
    # 6: cos(x) (low freq oscillation)
    # 7: exp(-x^2) (narrow Gaussian)
    # 8: exp(-x^2/2) (wide Gaussian)
    # 9: 1/(1+x^2) (Lorentzian/Rational)
    
    # Calculate basis functions
    # We use column_stack to create a matrix of shape (N, 10)
    basis = np.column_stack((
        np.ones_like(x),
        x,
        x**2,
        x**3,
        x**4,
        np.sin(x),
        np.cos(x),
        np.exp(-x**2),
        np.exp(-0.5 * x**2),
        1.0 / (1.0 + x**2)
    ))
    
    # Linear combination: y = basis @ params
    y = basis @ params
    
    return y
# EVOLVE-BLOCK-END
