"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Autonomous nonlinear oscillator model.
    We model acceleration a = f(x, v) as a polynomial in x and v.
    
    Physics prior:
    - No explicit time dependence (autonomous system).
    - Potential force: odd powers of x (cubic oscillator is common).
    - Damping: odd powers of v (linear, cubic).
    - Cross terms: x*v for parametric damping or coupling.
    
    Basis functions (10 params):
    0: 1
    1: x
    2: x^3
    3: x^5
    4: v
    5: v^3
    6: x*v
    7: x^2*v
    8: x*v^2
    9: v^2
    
    Note: x^5 and v^3 help capture strong nonlinearity.
    x^2*v and x*v^2 capture asymmetric damping.
    """
    p = params
    
    # Precompute powers to avoid redundant computation
    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    
    v2 = v * v
    v3 = v2 * v
    
    # Terms
    term0 = p[0]
    term1 = p[1] * x
    term2 = p[2] * x3
    term3 = p[3] * x5
    term4 = p[4] * v
    term5 = p[5] * v3
    term6 = p[6] * x * v
    term7 = p[7] * x2 * v
    term8 = p[8] * x * v2
    term9 = p[9] * v2
    
    return term0 + term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
# EVOLVE-BLOCK-END
