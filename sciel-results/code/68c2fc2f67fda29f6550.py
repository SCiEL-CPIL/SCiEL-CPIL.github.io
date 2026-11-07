"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a physics-informed basis.
    
    The model assumes the form of a driven Duffing/Van der Pol oscillator:
    a = p0 + p1*x + p2*x^3 + p3*v + p4*v^3 + p5*x*v + p6*sin(t) + p7*cos(t) + p8*t + p9
    
    This captures:
    - Linear restoring force (p1*x)
    - Nonlinear restoring force (p2*x^3)
    - Linear damping (p3*v)
    - Nonlinear damping (p4*v^3)
    - Coupling term (p5*x*v)
    - Periodic driving (p6*sin(t) + p7*cos(t))
    - Drift/linear time dependence (p8*t)
    - Constant offsets (p0, p9)
    
    Total 10 parameters.
    """
    # Unpack 10 parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute basis terms
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Trigonometric terms for periodic driving
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Construct the linear combination
    result = (p0 + 
              p1 * x + 
              p2 * x3 + 
              p3 * v + 
              p4 * v3 + 
              p5 * (x * v) + 
              p6 * sin_t + 
              p7 * cos_t + 
              p8 * t + 
              p9)
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(t)
        
    return result
# EVOLVE-BLOCK-END
