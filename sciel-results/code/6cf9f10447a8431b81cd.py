"""Symbolic regression: nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-inspired model for nonlinear oscillator acceleration.
    
    Combines:
    - Duffing-type position nonlinearity: p0*x + p1*x^3
    - Van der Pol-style velocity damping: p2*v + p3*v^3
    - Cross terms: p4*x*v + p5*x^2*v + p6*v^2
    - Time-varying terms (to capture any slow drift or phase): p7*sin(t) + p8*cos(t)
    - Constant offset: p9
    
    This structure is motivated by the known form of nonlinear oscillator equations:
    x'' = -ω²x - μ(x²-1)x' - βx³ + ...
    
    Input ranges:
      t in [30, 50]
      x in [-0.2056, 0.1749]
      v in [-0.3391, 0.3624]
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Core oscillator terms
    x_term = p0 * x + p1 * x**3
    v_term = p2 * v + p3 * v**3
    
    # Cross terms
    xv_term = p4 * x * v
    x2v_term = p5 * x**2 * v
    v2_term = p6 * v**2
    
    # Time-dependent terms (slow oscillation)
    t_term = p7 * np.sin(t) + p8 * np.cos(t)
    
    # Constant
    const = p9
    
    return x_term + v_term + xv_term + x2v_term + v2_term + t_term + const
# EVOLVE-BLOCK-END
