import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator.
    Uses a complete cubic polynomial basis in position (x) and velocity (v).
    Basis: 1, x, v, x^2, xv, v^2, x^3, x^2v, xv^2, v^3
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Compute powers
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Linear terms
    term1 = x
    term2 = v
    
    # Quadratic terms
    term3 = x2
    term4 = x * v
    term5 = v2
    
    # Cubic terms
    term6 = x3
    term7 = x2 * v
    term8 = x * v2
    term9 = v3
    
    # Combine
    return (p0 + p1 * term1 + p2 * term2 + p3 * term3 + p4 * term4 + 
            p5 * term5 + p6 * term6 + p7 * term7 + p8 * term8 + p9 * term9)
