"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator.
    Model: a = p0*t + p1*x + p2*v + p3 + p4*x^2 + p5*v^2 + p6*x*v + p7*t*x + p8*t*v + p9
    This captures linear dynamics, quadratic nonlinearities in position/velocity,
    and interactions with time.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Linear terms
    a = p0 * t + p1 * x + p2 * v + p3
    
    # Quadratic terms
    a = a + p4 * x**2 + p5 * v**2 + p6 * x * v
    
    # Time-interaction terms
    a = a + p7 * t * x + p8 * t * v
    
    # Constant offset (redundant with p3 if t is not centered, but provides flexibility)
    a = a + p9
    
    return a
# EVOLVE-BLOCK-END
