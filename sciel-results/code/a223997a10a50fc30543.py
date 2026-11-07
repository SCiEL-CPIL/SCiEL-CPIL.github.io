"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with driving force, from position and velocity.
    
    Uses a hybrid trigonometric-polynomial basis to capture periodic and nonlinear dynamics.
    Basis functions:
      sin(x), cos(x), sin(v), cos(v), x, v, x*v, sin(x)*v, x*sin(v), constant
    This approach leverages the periodic nature of oscillators, potentially offering better
    generalization than pure polynomials for nonlinear restoring and damping forces.
    """
    # Extract parameters
    p0 = params[0]  # coeff for sin(x)
    p1 = params[1]  # coeff for cos(x)
    p2 = params[2]  # coeff for sin(v)
    p3 = params[3]  # coeff for cos(v)
    p4 = params[4]  # coeff for x
    p5 = params[5]  # coeff for v
    p6 = params[6]  # coeff for x*v
    p7 = params[7]  # coeff for sin(x)*v
    p8 = params[8]  # coeff for x*sin(v)
    p9 = params[9]  # constant term
    
    # Compute trigonometric terms
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_v = np.sin(v)
    cos_v = np.cos(v)
    
    # Compute mixed terms
    x_v = x * v
    sin_x_v = sin_x * v
    x_sin_v = x * sin_v
    
    # Linear combination
    return p0 * sin_x + p1 * cos_x + p2 * sin_v + p3 * cos_v + \
           p4 * x + p5 * v + p6 * x_v + p7 * sin_x_v + p8 * x_sin_v + p9
# EVOLVE-BLOCK-END
