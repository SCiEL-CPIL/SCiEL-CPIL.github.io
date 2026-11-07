import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model using a polynomial basis.
    This approach uses a general cubic polynomial in position (x) and velocity (v)
    to capture smooth nonlinear dynamics.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Construct the polynomial equation
    # p0: 1 (bias)
    # p1: x
    # p2: v
    # p3: x^2
    # p4: v^2
    # p5: x * v
    # p6: x^3
    # p7: v^3
    # p8: x^2 * v
    # p9: x * v^2
    
    return (p0 + 
            p1 * x + 
            p2 * v + 
            p3 * x**2 + 
            p4 * v**2 + 
            p5 * x * v + 
            p6 * x**3 + 
            p7 * v**3 + 
            p8 * x**2 * v + 
            p9 * x * v**2)
