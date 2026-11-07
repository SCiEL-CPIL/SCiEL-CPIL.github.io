import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rich basis: polynomial terms (up to degree 5), trigonometric terms,
    tanh saturating terms, and exponential terms. All with parametric scales.
    
    10 parameters:
    0: polynomial coefficient a0
    1: polynomial coefficient a1
    2: polynomial coefficient a2
    3: polynomial coefficient a3
    4: polynomial coefficient a4
    5: trig frequency f
    6: trig amplitude A
    7: tanh scale s
    8: tanh amplitude B
    9: exp scale g
    """
    t0 = time.time()
    
    # Safety: check for non-finite params
    if not np.all(np.isfinite(params)):
        return np.zeros_like(x)
    
    a0, a1, a2, a3, a4, f, A, s, B, g = params
    
    # Clip parameters to avoid numerical issues
    f = np.clip(f, -20.0, 20.0)
    s = np.clip(s, 0.01, 10.0)
    g = np.clip(g, -5.0, 5.0)
    
    # Polynomial terms (degree 0-4)
    y = a0 * np.ones_like(x)
    y += a1 * x
    y += a2 * x**2
    y += a3 * x**3
    y += a4 * x**4
    
    # Trigonometric term: A * sin(f * x)
    y += A * np.sin(f * x)
    
    # Tanh saturating term: B * tanh(s * x)
    y += B * np.tanh(s * x)
    
    # Exponential term: np.exp(g * x) - 1 (centered at x=0)
    exp_term = np.exp(np.clip(g * x, -50.0, 50.0)) - 1.0
    y += exp_term
    
    # Safety: check for non-finite values
    if not np.all(np.isfinite(y)):
        # Fallback to just polynomial part
        y = a0 * np.ones_like(x)
        y += a1 * x
        y += a2 * x**2
        y += a3 * x**3
        y += a4 * x**4
        if not np.all(np.isfinite(y)):
            return np.zeros_like(x)
    
    return y
