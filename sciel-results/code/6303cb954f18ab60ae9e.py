"""Symbolic regression with Chebyshev + RBF hybrid basis for better conditioning."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a hybrid Chebyshev + RBF basis.
    
    Basis functions (10 total):
    1-4: Chebyshev polynomials T0-T3 mapped to [-1,1] (well-conditioned)
    5-6: sin(x), cos(x) for low-frequency periodic components
    7-8: sin(2x), cos(2x) for medium-frequency periodic components
    9:   Gaussian RBF centered at 0 (localized bump)
    10:  x (linear term for global trend)
    
    This basis is well-conditioned and covers polynomials, periodic,
    and localized features of the hidden function.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    n = x.shape[0]
    
    # Map to [-1, 1] for Chebyshev polynomials
    t = x / 3.0
    
    # Chebyshev polynomials T0-T3
    T0 = np.ones(n)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    
    # Sinusoidal components
    S1 = np.sin(x)
    C1 = np.cos(x)
    S2 = np.sin(2.0 * x)
    C2 = np.cos(2.0 * x)
    
    # Localized Gaussian bump
    G0 = np.exp(-x**2)
    
    # Linear term
    L = x
    
    # Combine all basis functions
    result = (params[0] * T0 + 
              params[1] * T1 + 
              params[2] * T2 + 
              params[3] * T3 + 
              params[4] * S1 + 
              params[5] * C1 + 
              params[6] * S2 + 
              params[7] * C2 + 
              params[8] * G0 + 
              params[9] * L)
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.zeros_like(x)
    
    return result
# EVOLVE-BLOCK-END
