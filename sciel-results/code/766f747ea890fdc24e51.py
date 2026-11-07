import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model using a polynomial basis.
    This approach uses a cubic polynomial in position (x) and velocity (v)
    to capture nonlinear dynamics such as stiffness and damping effects.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute powers
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    
    # Construct the equation
    # p0: constant
    # p1: x
    # p2: v
    # p3: x^2
    # p4: v^2
    # p5: x*v
    # p6: x^3
    # p7: v^3
    # p8: x^2*v
    # p9: x*v^2
    
    return (p0 + 
            p1 * x + 
            p2 * v + 
            p3 * x2 + 
            p4 * v2 + 
            p5 * x * v + 
            p6 * x3 + 
            p7 * v3 + 
            p8 * x2 * v + 
            p9 * x * v2)
