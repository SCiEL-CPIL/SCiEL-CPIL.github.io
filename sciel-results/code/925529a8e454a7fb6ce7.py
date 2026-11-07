import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter basis using a full degree-3 polynomial basis.
    This is a structural change from the hybrid polynomial/trig/tanh basis
    to a pure polynomial basis, which may better capture smooth functions
    on [-1,1]^2 without the nonlinearity constraints of trig/tanh functions.
    
    Basis (10 terms for full degree-3 in 2D):
    1, x, y, x^2, xy, y^2, x^3, x^2*y, x*y^2, y^3
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Degree 0
    term_0 = np.ones_like(x)
    
    # Degree 1
    term_1 = x
    term_2 = y
    
    # Degree 2
    term_3 = x**2
    term_4 = x * y
    term_5 = y**2
    
    # Degree 3
    term_6 = x**3
    term_7 = x**2 * y
    term_8 = x * y**2
    term_9 = y**3
    
    return (p0 * term_0 
            + p1 * term_1 
            + p2 * term_2 
            + p3 * term_3 
            + p4 * term_4 
            + p5 * term_5 
            + p6 * term_6 
            + p7 * term_7 
            + p8 * term_8 
            + p9 * term_9)
