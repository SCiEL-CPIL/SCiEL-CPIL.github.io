"""Symbolic regression: nonlinear oscillator (2) - closed-form equation."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator (Duffing-type) with sinusoidal driving.
    
    Model: a = c1*t + c2*x + c3*v + c4*x^3 + c5*v^3 + c6*sin(w0*t) + c7*x^2 + c8*v^2 + c9*x*v + c10
    
    This captures:
    - Linear restoring and damping
    - Cubic stiffness (Duffing nonlinearity)
    - Cubic velocity damping
    - Sinusoidal driving force
    - Quadratic correction terms
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Ensure finite computation
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Base frequency for driving term (use a reasonable default, let params scale it)
    # We use a fixed frequency of 1.0 rad/s for the sinusoidal drive, scaled by param
    drive = np.sin(1.0 * t)
    
    result = (
        p0 * t +
        p1 * x +
        p2 * v +
        p3 * x3 +
        p4 * v3 +
        p5 * drive +
        p6 * x2 +
        p7 * v2 +
        p8 * x * v +
        p9
    )
    
    return result
# EVOLVE-BLOCK-END
