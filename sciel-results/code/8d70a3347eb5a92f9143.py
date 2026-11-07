"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Chebyshev polynomials (global smooth trend) + Fourier terms (oscillations).
    
    The 10 parameters weight:
      0-3: Chebyshev T_0 to T_3 on normalized x in [-1, 1]
      4-9: sin/cos at frequencies 1, 2, 3
    
    This is linear in parameters, so BFGS finds the global least-squares optimum.
    """
    start_time = time.time()
    
    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    
    if n == 0:
        return np.array([])
    
    # Normalize to [-1, 1] for Chebyshev stability (x is in [-3, 3])
    t = x / 3.0
    
    # Chebyshev polynomials T_0 to T_3
    b0 = np.ones(n)
    b1 = t.copy()
    b2 = 2.0 * t**2 - 1.0
    b3 = 4.0 * t**3 - 3.0 * t
    
    # Fourier terms
    b4 = np.sin(x)
    b5 = np.cos(x)
    b6 = np.sin(2.0 * x)
    b7 = np.cos(2.0 * x)
    b8 = np.sin(3.0 * x)
    b9 = np.cos(3.0 * x)
    
    # Stack and compute linear combination
    basis = np.vstack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    result = basis.T @ params
    
    # Safety: ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
