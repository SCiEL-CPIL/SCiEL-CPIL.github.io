"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator with Duffing, Van der Pol, and sinusoidal driving.
    Basis:
    p0: offset
    p1: linear spring (x)
    p2: quadratic spring (x^2)
    p3: cubic spring (x^3)
    p4: linear damping (v)
    p5: quadratic damping (v^2)
    p6: cubic damping (v^3)
    p7: sine drive (sin(t))
    p8: cosine drive (cos(t))
    p9: coupling (x*v)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    st = np.sin(t)
    ct = np.cos(t)
    
    return p0 + p1*x + p2*x2 + p3*x3 + p4*v + p5*v2 + p6*v3 + p7*st + p8*ct + p9*xv
# EVOLVE-BLOCK-END
