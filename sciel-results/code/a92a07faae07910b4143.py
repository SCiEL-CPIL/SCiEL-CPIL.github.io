import numpy as np
import time

# Deadline guard: kill after 140s (10s margin before 150s limit)
_DEADLINE = time.time() + 140.0

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-informed Duffing-van der Pol structure for a nonlinear oscillator.
    
    a = p1*x + p2*x^3 - p3*v - p4*v^3 + p5*x*v^2 + p6*v^2*x^2 + p7*x^2*v + p8*x^4 + p9*v^4 + p10
    
    This captures:
    - Linear restoring force (p1*x)
    - Cubic nonlinearity (p2*x^3) - Duffing term
    - Linear damping (p3*v)
    - Cubic damping (p4*v^3) - van der Pol type
    - Cross terms coupling position and velocity
    - Higher order terms for accuracy
    """
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = params
    
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    
    result = (p1 * x + 
              p2 * x3 + 
              p3 * v + 
              p4 * v3 + 
              p5 * x * v2 + 
              p6 * v2 * x2 + 
              p7 * x2 * v + 
              p8 * x4 + 
              p9 * v4 + 
              p10)
    
    return result
