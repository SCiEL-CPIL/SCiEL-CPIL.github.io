"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = None
_TIME_LIMIT = 115.0  # Safety margin before 120s kill


def _check_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT
    return time.time() < _DEADLINE


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximate the target function using a Fourier series basis.
    
    The Fourier basis {1, sin(k*pi*x/L), cos(k*pi*x/L)} is orthogonal on [0, L],
    providing a well-conditioned basis for BFGS optimization. We use L=6 to match
    the upper bound of the domain [0.1, 6].
    
    Args:
        x: Input array of shape (N,).
        params: Array of 10 parameters.
        
    Returns:
        Prediction array of shape (N,).
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Check deadline
    if not _check_deadline():
        return np.ones_like(x) * params[0]
    
    # Domain length for Fourier basis
    L = 6.0
    
    # Compute Fourier basis: 1, sin(pi*x/L), cos(pi*x/L), sin(2*pi*x/L), cos(2*pi*x/L), ...
    # We have 10 parameters, so: 1 + 9 more terms
    # Pattern: 1, sin(1), cos(1), sin(2), cos(2), sin(3), cos(3), sin(4), cos(4)
    # That's 1 + 8 = 9 terms... we need 10.
    # Let's use: 1, sin(1), cos(1), sin(2), cos(2), sin(3), cos(3), sin(4), cos(4), sin(5)
    
    n = len(x)
    basis = np.empty((n, 10), dtype=np.float64)
    
    # Term 0: constant
    basis[:, 0] = 1.0
    
    # Terms 1-9: alternating sin and cos
    # Term 1: sin(pi*x/L)
    # Term 2: cos(pi*x/L)
    # Term 3: sin(2*pi*x/L)
    # Term 4: cos(2*pi*x/L)
    # Term 5: sin(3*pi*x/L)
    # Term 6: cos(3*pi*x/L)
    # Term 7: sin(4*pi*x/L)
    # Term 8: cos(4*pi*x/L)
    # Term 9: sin(5*pi*x/L)
    
    for k in range(1, 10):
        if k % 2 == 1:
            # Odd k: sin(k*pi*x/L)
            idx = k  # param index k (1-based)
            basis[:, idx] = np.sin(k * np.pi * x / L)
        else:
            # Even k: cos(k*pi*x/L)
            idx = k  # param index k (1-based)
            basis[:, idx] = np.cos(k * np.pi * x / L)
    
    # Check for non-finite values
    if not np.all(np.isfinite(basis)):
        basis = np.nan_to_num(basis, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Prediction is the linear combination of the basis functions
    result = basis @ params
    
    # Ensure output is finite
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return result
# EVOLVE-BLOCK-END
