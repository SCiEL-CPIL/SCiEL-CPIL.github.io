"""Rewritten program for symbolic regression of a nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with driving force.
    
    Uses a cubic polynomial basis in x and v, structured to match physical oscillator terms:
    - Constant (driving offset)
    - Linear in x (restoring force)
    - Linear in v (damping)
    - Quadratic in x (nonlinear stiffness)
    - Cross term x*v
    - Quadratic in v (nonlinear damping)
    - Cubic in x (strong nonlinearity)
    - x^2*v
    - x*v^2
    - Cubic in v
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    x2v = x2 * v
    xv2 = x * v2
    
    return (p0 + p1*x + p2*v + p3*x2 + p4*xv + p5*v2 + 
            p6*x3 + p7*x2v + p8*xv2 + p9*v3)
# EVOLVE-BLOCK-END
