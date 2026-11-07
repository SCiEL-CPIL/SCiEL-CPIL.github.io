"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for the acceleration of a nonlinear oscillator.
    
    Inputs:
        t: time
        x: position
        v: velocity
        params: array of 10 fitted constants
    
    Uses a basis of 10 terms spanning:
        constant, t, x, v, x^2, x^3, v^2, x*v, sin(t), cos(t)
    
    This allows BFGS to fit a rich nonlinear oscillator model including:
        - linear damping (v term)
        - nonlinear stiffness (x, x^2, x^3)
        - velocity-squared damping (v^2)
        - cross-coupling (x*v)
        - periodic driving (sin(t), cos(t))
        - linear time trend (t)
        - constant offset
    """
    return (
        params[0]
        + params[1] * t
        + params[2] * x
        + params[3] * v
        + params[4] * x * x
        + params[5] * x * x * x
        + params[6] * v * v
        + params[7] * x * v
        + params[8] * np.sin(t)
        + params[9] * np.cos(t)
    )
# EVOLVE-BLOCK-END
