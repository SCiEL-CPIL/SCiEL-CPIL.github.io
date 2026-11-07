"""Nonlinear oscillator equation with structural improvements."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear driven-damped oscillator.
    
    Model: a = c1*t + c2*x + c3*v + c4 + c5*x^3 + c6*v^3 
             + c7*cos(w*t + phi)*A + c8*x*v + c9*x^2 + c10*v^2
    
    Parameters:
    - p[0]: linear time coefficient (driving)
    - p[1]: linear spring coefficient
    - p[2]: linear damping coefficient
    - p[3]: constant offset
    - p[4]: cubic spring coefficient
    - p[5]: cubic damping coefficient
    - p[6]: sinusoidal drive amplitude
    - p[7]: sinusoidal drive frequency
    - p[8]: sinusoidal drive phase
    - p[9]: cross-term x*v coefficient
    """
    p = params
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = p
    
    # Linear terms
    linear = c1 * t + c2 * x + c3 * v + c4
    
    # Nonlinear spring
    nonlinear_spring = c5 * x**3
    
    # Nonlinear damping
    nonlinear_damping = c6 * v**3
    
    # Sinusoidal time-dependent drive
    drive = c7 * np.cos(c8 * t + c9)
    
    # Cross term
    cross = c10 * x * v
    
    return linear + nonlinear_spring + nonlinear_damping + drive + cross
# EVOLVE-BLOCK-END
