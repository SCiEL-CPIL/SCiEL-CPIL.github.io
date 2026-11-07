import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using cubic polynomial + two Gaussian bumps.
    
    Well-conditioned for BFGS:
    - Cubic polynomial captures global trend
    - Two Gaussian bumps capture localized features
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    
    a1 = params[4]
    s1 = params[5]
    c1 = params[6]
    
    a2 = params[7]
    s2 = params[8]
    c2 = params[9]
    
    # Polynomial part
    y = p0 + p1 * x + p2 * x * x + p3 * x * x * x
    
    # Gaussian bumps with variance-like parameter (better conditioned)
    s1_safe = np.maximum(s1, 0.01)
    s2_safe = np.maximum(s2, 0.01)
    
    y += a1 * np.exp(-(x - c1) ** 2 / s1_safe)
    y += a2 * np.exp(-(x - c2) ** 2 / s2_safe)
    
    return y
