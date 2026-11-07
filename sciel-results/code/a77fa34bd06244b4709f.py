import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Chebyshev polynomials of the first kind
    # T0 = 1
    # T1 = x
    # T2 = 2x^2 - 1
    # T3 = 4x^3 - 3x
    
    T0_x = 1.0
    T1_x = x
    T2_x = 2.0 * x**2 - 1.0
    T3_x = 4.0 * x**3 - 3.0 * x
    
    T0_y = 1.0
    T1_y = y
    T2_y = 2.0 * y**2 - 1.0
    T3_y = 4.0 * y**3 - 3.0 * y
    
    # Tensor product basis up to degree 3
    # Total 10 terms:
    # 0: T0_x * T0_y
    # 1: T1_x * T0_y
    # 2: T0_x * T1_y
    # 3: T2_x * T0_y
    # 4: T0_x * T2_y
    # 5: T1_x * T1_y
    # 6: T3_x * T0_y
    # 7: T0_x * T3_y
    # 8: T2_x * T1_y
    # 9: T1_x * T2_y
    
    return (p0 * T0_x * T0_y + 
            p1 * T1_x * T0_y + 
            p2 * T0_x * T1_y + 
            p3 * T2_x * T0_y + 
            p4 * T0_x * T2_y + 
            p5 * T1_x * T1_y + 
            p6 * T3_x * T0_y + 
            p7 * T0_x * T3_y + 
            p8 * T2_x * T1_y + 
            p9 * T1_x * T2_y)
