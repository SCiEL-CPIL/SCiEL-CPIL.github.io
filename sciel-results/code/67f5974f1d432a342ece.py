import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Chebyshev product basis T_i(x)T_j(y) 
    for total degree i+j <= 3.
    
    Basis terms:
    0: T0(x)T0(y) = 1
    1: T1(x)T0(y) = x
    2: T0(x)T1(y) = y
    3: T2(x)T0(y) = 2x^2 - 1
    4: T1(x)T1(y) = xy
    5: T0(x)T2(y) = 2y^2 - 1
    6: T3(x)T0(y) = 4x^3 - 3x
    7: T2(x)T1(y) = (2x^2 - 1)y
    8: T1(x)T2(y) = x(2y^2 - 1)
    9: T0(x)T3(y) = 4y^3 - 3y
    """
    # Compute Chebyshev polynomials up to degree 3 for x and y
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # T3(t) = 4t^3 - 3t
    
    x2 = x * x
    x3 = x2 * x
    y2 = y * y
    y3 = y2 * y
    
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x2 - 1
    T3_x = 4 * x3 - 3 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y2 - 1
    T3_y = 4 * y3 - 3 * y
    
    # Construct the basis functions
    # Order: (i, j) where i is degree in x, j is degree in y
    # 0: (0,0)
    # 1: (1,0)
    # 2: (0,1)
    # 3: (2,0)
    # 4: (1,1)
    # 5: (0,2)
    # 6: (3,0)
    # 7: (2,1)
    # 8: (1,2)
    # 9: (0,3)
    
    b0 = T0_x * T0_y
    b1 = T1_x * T0_y
    b2 = T0_x * T1_y
    b3 = T2_x * T0_y
    b4 = T1_x * T1_y
    b5 = T0_x * T2_y
    b6 = T3_x * T0_y
    b7 = T2_x * T1_y
    b8 = T1_x * T2_y
    b9 = T0_x * T3_y
    
    pred = (params[0] * b0 + 
            params[1] * b1 + 
            params[2] * b2 + 
            params[3] * b3 + 
            params[4] * b4 + 
            params[5] * b5 + 
            params[6] * b6 + 
            params[7] * b7 + 
            params[8] * b8 + 
            params[9] * b9)
            
    return pred
