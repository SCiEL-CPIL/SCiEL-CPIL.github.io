"""Symbolic regression: nonlinear oscillator (2) - closed-form equation."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with time-dependent driving force.
    
    Uses a physically-motivated nonlinear basis:
    - Linear restoring force: x
    - Cubic stiffness (Duffing): x^3
    - Linear damping: v
    - Velocity nonlinearity: v^2, v^3
    - Time-dependent driving: sin(t), cos(t), t, t^2
    - Cross terms: x*v, t*x, t*v
    
    10 fitted constants.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Core terms
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Time-dependent terms
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    t2 = t * t
    
    # Cross terms
    xv = x * v
    tx = t * x
    tv = t * v
    
    # Combine
    result = (p0 * x + 
              p1 * x3 + 
              p2 * v + 
              p3 * v2 + 
              p4 * sin_t + 
              p5 * cos_t + 
              p6 * xv + 
              p7 * t + 
              p8 * t2 + 
              p9)
    
    return result
# EVOLVE-BLOCK-END
