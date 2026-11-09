"""Symbolic regression: nonlinear oscillator equation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for a nonlinear damped oscillator with driving force.
    
    Structure:
    a = p0*x + p1*x^3 + p2*v + p3*v^3 + p4*x*v + p5*sin(t) + p6*cos(t) 
        + p7*sin(2*t) + p8*cos(2*t) + p9
    
    Inputs:
        t: time
        x: position
        v: velocity
        params: array of 10 fitted constants
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Nonlinear restoring force (cubic stiffness)
    restoring = p0 * x + p1 * x**3
    
    # Damping (linear + cubic)
    damping = p2 * v + p3 * v**3
    
    # Cross term (nonlinear coupling)
    cross = p4 * x * v
    
    # Periodic driving force with fundamental and 2nd harmonic
    drive = p5 * np.sin(t) + p6 * np.cos(t) + p7 * np.sin(2.0 * t) + p8 * np.cos(2.0 * t)
    
    # Constant offset
    const = p9
    
    return restoring + damping + cross + drive + const
# EVOLVE-BLOCK-END
