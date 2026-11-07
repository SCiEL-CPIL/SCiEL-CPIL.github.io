"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with driving force, from position and velocity.
    
    Uses a physically-motivated Van der Pol + Duffing structure.
    The (x^2 - 1)*v term explicitly encodes the limit cycle mechanism
    (negative damping at small amplitudes, positive at large).
    
    Basis terms:
      1, x, v, x^2, x*v, v^2, (x^2-1)*v, x^3, x^2*v, x*v^2
    
    This captures:
      - Linear spring and damping
      - Quadratic nonlinearities
      - Cubic Duffing stiffness (x^3)
      - Van der Pol limit cycle damping ((x^2-1)*v)
      - Velocity nonlinearities
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    v2 = v * v
    x3 = x2 * x
    x2v = x2 * v
    xv2 = x * v2
    
    return (p0 * x + 
            p1 * v + 
            p2 * x2 + 
            p3 * x * v + 
            p4 * v2 + 
            p5 * (x2 - 1.0) * v + 
            p6 * x3 + 
            p7 * x2v + 
            p8 * xv2 + 
            p9)
# EVOLVE-BLOCK-END
