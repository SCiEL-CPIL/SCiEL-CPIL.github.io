"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model with hyperbolic nonlinearities.
    
    Model:
    a = p0 * sin(p1 * t + p2) + p3 * tanh(p4 * x) + p5 * tanh(p6 * v) + p7 * x * v + p8 * x**2 + p9
    
    Parameters:
    p0: Drive amplitude
    p1: Drive frequency
    p2: Drive phase
    p3: Nonlinear position strength
    p4: Nonlinear position scale
    p5: Nonlinear velocity strength
    p6: Nonlinear velocity scale
    p7: Velocity-position coupling
    p8: Quadratic position term
    p9: Constant bias
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Driving force
    drive = p0 * np.sin(p1 * t + p2)
    
    # Nonlinear restoring and damping forces (Hyperbolic)
    nonlinear_restoring = p3 * np.tanh(p4 * x)
    nonlinear_damping = p5 * np.tanh(p6 * v)
    
    # Interaction and other terms
    interaction = p7 * x * v
    quadratic_vel = p8 * v**2
    
    # Total acceleration
    return drive + nonlinear_restoring + nonlinear_damping + interaction + quadratic_vel + p9
# EVOLVE-BLOCK-END
