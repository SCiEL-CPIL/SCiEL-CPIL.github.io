"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with driving force, from position and velocity."""
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Trigonometric basis terms to capture periodic oscillatory behavior
    # This differs structurally from pure polynomial bases
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_v = np.sin(v)
    cos_v = np.cos(v)
    
    # Nonlinear polynomial terms for Duffing-type behavior
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Cross terms
    xv = x * v
    x2v = x2 * v
    xv2 = x * v2
    
    # Mix: use trig for periodicity, polynomial for nonlinearity
    # p0: constant offset
    # p1: linear x (restoring)
    # p2: linear v (damping)
    # p3: sin(x) (periodic restoring)
    # p4: cos(x) * v (parametric coupling)
    # p5: x^3 (Duffing cubic)
    # p6: v^3 (nonlinear damping)
    # p7: x*v (cross damping)
    # p8: sin(v) (velocity oscillation component)
    # p9: x^2 (quadratic nonlinearity)
    
    return (p0 + p1 * x + p2 * v + p3 * sin_x + p4 * cos_x * v +
            p5 * x3 + p6 * v3 + p7 * xv + p8 * sin_v + p9 * x2)
# EVOLVE-BLOCK-END
