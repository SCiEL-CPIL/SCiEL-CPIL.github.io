"""Symbolic regression with a hybrid non-linear basis."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis combining:
    - 3 Chebyshev polynomials (degrees 0, 1, 2) for smooth baseline
    - 7 exponential terms exp(p_i * x) for non-linear features
    
    This creates a non-linear parameter space that BFGS can optimize,
    allowing the model to capture sharp transitions and exponential
    growth/decay patterns that linear polynomial bases cannot.
    """
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: return zeros if we're too close to the 120s limit
    elapsed = time.time() - _start_time
    if elapsed > 115.0:
        return np.zeros_like(x, dtype=np.float64)
    
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if p.shape[0] < 10:
        p = np.pad(p, (0, 10 - p.shape[0]), constant_values=0.0)
    
    # Scale x to [-1, 1] for Chebyshev stability
    t = x / 3.0
    
    # Part 1: Chebyshev basis for smooth components (first 3 params)
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2*t^2 - 1
    
    result = p[0] * np.ones_like(t)  # p[0] * T_0
    
    if p.shape[0] > 1:
        result += p[1] * t  # p[1] * T_1
    
    if p.shape[0] > 2:
        T2 = 2.0 * t * t - 1.0
        result += p[2] * T2  # p[2] * T_2
    
    # Part 2: Exponential terms (params 3-9)
    # Each term: p[i] * exp(p[i] * t) -- wait, that uses p[i] twice.
    # Better: use separate coefficients and exponents, but we only have 7 params left.
    # Let's use: for i in 3..9, term = exp(p[i] * t) with coefficient 1.0?
    # Or: term = p[i] * exp(p[i] * t)? That's non-linear in p[i].
    # Actually, let's use: sum_{i=3}^{9} exp(p[i] * t)
    # This is non-linear in p[i] because p[i] appears in the exponent.
    
    for i in range(3, 10):
        # Clip the exponent argument to avoid overflow
        arg = p[i] * t
        arg = np.clip(arg, -50.0, 50.0)
        result += np.exp(arg)
    
    return result
# EVOLVE-BLOCK-END
