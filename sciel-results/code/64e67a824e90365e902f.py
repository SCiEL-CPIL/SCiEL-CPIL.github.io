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
    Predicts y using a flexible basis function approach.
    x: input array
    params: array of 10 coefficients
    Returns: predictions array same shape as x
    """
    if not _elapsed_ok():
        return np.zeros_like(x)
    
    # Ensure x is float for operations
    x = np.asarray(x, dtype=float)
    
    # Scale x to [-1, 1] for better numerical conditioning
    t = x / 3.0
    
    # Construct a diverse, well-conditioned basis:
    # Use Chebyshev polynomials (well-conditioned on [-1,1]) combined with
    # a few smooth non-polynomial terms for flexibility
    # 
    # Chebyshev polynomials T_0 through T_5 give 6 well-conditioned polynomial terms
    # Plus: sin(pi*t), cos(pi*t), exp(t), tanh(t) for non-polynomial flexibility
    # Total: 10 basis functions
    
    # Chebyshev polynomials of the first kind
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2 * t**2 - 1
    T3 = 4 * t**3 - 3 * t
    T4 = 8 * t**4 - 8 * t**2 + 1
    T5 = 16 * t**5 - 20 * t**3 + 5 * t
    
    # Non-polynomial terms (all smooth and bounded on [-1,1])
    S1 = np.sin(np.pi * t)
    C1 = np.cos(np.pi * t)
    E1 = np.exp(t)  # bounded on [e^-1, e] ≈ [0.37, 2.72]
    H1 = np.tanh(t)  # bounded on [-tanh(1), tanh(1)] ≈ [-0.76, 0.76]
    
    # Stack basis functions
    basis = np.column_stack((
        T0, T1, T2, T3, T4, T5,
        S1, C1, E1, H1
    ))
    
    # Linear combination: y = basis @ params
    y = basis @ params
    
    return y
# EVOLVE-BLOCK-END
