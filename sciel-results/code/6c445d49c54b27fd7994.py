import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a complete cubic polynomial basis.
    Terms:
    p0: 1
    p1: v
    p2: v^2
    p3: v^3
    p4: x
    p5: x*v
    p6: x*v^2
    p7: x^2
    p8: x^2*v
    p9: x^3
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    term_0 = p0
    term_1 = p1 * v
    term_2 = p2 * v**2
    term_3 = p3 * v**3
    term_4 = p4 * x
    term_5 = p5 * x * v
    term_6 = p6 * x * v**2
    term_7 = p7 * x**2
    term_8 = p8 * x**2 * v
    term_9 = p9 * x**3
    
    return term_0 + term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8 + term_9
