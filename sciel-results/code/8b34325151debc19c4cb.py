"""Symbolic regression: nonlinear oscillator (2) - closed-form equation."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with time-dependent driving force.
    
    Model:
      a = c0 + c1*t + c2*x + c3*v + c4*x^2 + c5*x^3 + c6*v^2 + c7*x*v + c8*t*x + c9*t*v
    
    This captures:
      - Constant offset (c0)
      - Time-dependent driving (c1*t, c8*t*x, c9*t*v)
      - Linear restoring force (c2*x)
      - Linear damping (c3*v)
      - Nonlinear stiffness (c4*x^2, c5*x^3)
      - Velocity nonlinearity (c6*v^2)
      - Cross term between displacement and velocity (c7*x*v)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure inputs are arrays
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    
    # Compute basis terms
    t2 = t * t
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    xv = x * v
    tx = t * x
    tv = t * v
    
    # Combine with parameters
    result = (p0 
              + p1 * t 
              + p2 * x 
              + p3 * v 
              + p4 * x2 
              + p5 * x3 
              + p6 * v2 
              + p7 * xv 
              + p8 * tx 
              + p9 * tv)
    
    return result
# EVOLVE-BLOCK-END
