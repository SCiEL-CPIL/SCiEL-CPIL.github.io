"""Rewritten program using a hybrid Exponential-Sinusoidal-Polynomial basis."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model:
    y = A1 * exp(B1 * t) + A2 * exp(B2 * t) + C * sin(D * t + E) + F * t^2 + G * t + H
    
    where t = x / 2.0 to scale input to [-1, 1].
    
    params:
    0: A1 (amp exp 1)
    1: B1 (rate exp 1)
    2: A2 (amp exp 2)
    3: B2 (rate exp 2)
    4: C  (amp sin)
    5: D  (freq sin)
    6: E  (phase sin)
    7: F  (coeff t^2)
    8: G  (coeff t)
    9: H  (bias)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([], dtype=float)
    
    # Scale input to [-1, 1] for numerical stability of exponentials
    t = x / 2.0
    
    # Extract parameters
    A1, B1, A2, B2, C, D, E, F, G, H = params
    
    # Calculate components
    # Clip exponent arguments to prevent overflow/underflow issues during optimization
    # exp(30) is ~1e13, exp(-30) is ~1e-13. This is safe for float64.
    exp_arg1 = np.clip(B1 * t, -30, 30)
    exp_arg2 = np.clip(B2 * t, -30, 30)
    
    term_exp1 = A1 * np.exp(exp_arg1)
    term_exp2 = A2 * np.exp(exp_arg2)
    
    term_sin = C * np.sin(D * t + E)
    
    term_poly = F * (t**2) + G * t + H
    
    y = term_exp1 + term_exp2 + term_sin + term_poly
    
    return y
# EVOLVE-BLOCK-END
