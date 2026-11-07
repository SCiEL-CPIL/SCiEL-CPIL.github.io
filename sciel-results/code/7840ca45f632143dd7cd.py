"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator (Van der Pol type).
    
    Uses a physics-inspired basis that captures:
    - Linear restoring force (x)
    - Linear damping (v)
    - Nonlinear stiffness (x^3, x^5)
    - Amplitude-dependent damping (v^2, v^3, x^2*v)
    - Cross terms (x*v, x*v^2)
    - Constant offset
    
    Basis: {1, x, v, x^2, x*v, v^2, x^3, x^2*v, x*v^2, v^3}
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Build the prediction as a weighted sum of basis functions
    pred = (p0 * x + 
            p1 * v + 
            p2 * x2 + 
            p3 * x * v + 
            p4 * v2 + 
            p5 * x3 + 
            p6 * x2 * v + 
            p7 * x * v2 + 
            p8 * v3 + 
            p9)
    
    return pred
# EVOLVE-BLOCK-END
