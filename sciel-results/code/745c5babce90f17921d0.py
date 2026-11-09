import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a physics-informed polynomial basis.
    Includes Duffing (x^3) and Van der Pol (x^2*v) terms, plus other interactions.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Core physics terms
    term_x = p1 * x
    term_v = p2 * v
    term_duffing = p4 * x**3
    term_vdp_damping = p6 * x**2 * v
    
    # Additional polynomial terms for flexibility
    term_const = p0
    term_x2 = p3 * x**2
    term_xv = p5 * x * v
    term_v2 = p7 * v**2
    term_x4 = p8 * x**4
    term_xv2 = p9 * x * v**2
    
    return term_const + term_x + term_v + term_x2 + term_duffing + term_xv + term_vdp_damping + term_v2 + term_x4 + term_xv2
