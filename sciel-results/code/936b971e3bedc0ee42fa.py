"""Symbolic regression: equation for a nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form acceleration for a damped nonlinear oscillator.
    
    Uses a polynomial basis with up to 10 fitted constants:
    a = p0*x + p1*v + p2*x^2 + p3*x*v + p4*v^2 + p5*x^3 + p6*x^2*v + p7*v^3 + p8*x^2 + p9
    
    The basis is chosen to capture common nonlinear oscillator terms:
    - Linear: x, v
    - Quadratic: x^2, x*v, v^2
    - Cubic: x^3, x^2*v, v^3
    - Additional: x^2 (redundant with quadratic but helps fitting), constant
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    x2v = x2 * v
    
    return (p0 * x + p1 * v + p2 * x2 + p3 * xv + p4 * v2 +
            p5 * x3 + p6 * x2v + p7 * v3 + p8 * x2 + p9)
# EVOLVE-BLOCK-END
