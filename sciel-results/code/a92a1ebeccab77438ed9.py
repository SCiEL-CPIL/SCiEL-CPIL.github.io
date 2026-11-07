import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter basis using Chebyshev polynomials for symbolic regression on [-1,1] x [-1,1].
    
    Basis:
    1. 1
    2. x
    3. y
    4. T2(x) = 2x^2 - 1
    5. T2(y) = 2y^2 - 1
    6. x * y
    7. T3(x) = 4x^3 - 3x
    8. T3(y) = 4y^3 - 3y
    9. x * T2(y)
    10. y * T2(x)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Chebyshev polynomials
    T2_x = 2 * x**2 - 1
    T2_y = 2 * y**2 - 1
    T3_x = 4 * x**3 - 3 * x
    T3_y = 4 * y**3 - 3 * y
    
    return (p0 
            + p1 * x 
            + p2 * y 
            + p3 * T2_x 
            + p4 * T2_y 
            + p5 * x * y 
            + p6 * T3_x 
            + p7 * T3_y 
            + p8 * x * T2_y 
            + p9 * y * T2_x)
