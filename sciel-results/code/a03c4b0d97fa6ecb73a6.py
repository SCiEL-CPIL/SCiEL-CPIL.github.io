import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a 3rd-order polynomial basis.
    Uses raw x and v (no normalization) to avoid conditioning issues with BFGS.
    
    Basis: All monomials x^i * v^j where i + j <= 3.
    This captures Duffing-like terms (x^3), drag-like terms (xv, v^2), etc.
    """
    p = params
    
    # Compute powers
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Degree 0
    term0 = p[0]
    
    # Degree 1
    term1 = p[1] * x + p[2] * v
    
    # Degree 2
    term2 = p[3] * x2 + p[4] * x * v + p[5] * v2
    
    # Degree 3
    term3 = p[6] * x3 + p[7] * x2 * v + p[8] * x * v2 + p[9] * v3
    
    return term0 + term1 + term2 + term3
