import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Chebyshev polynomials up to T3
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2 * x**2 - 1
    T3_x = 4 * x**3 - 3 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2 * y**2 - 1
    T3_y = 4 * y**3 - 3 * y
    
    # Basis functions: Separable tensor-product basis
    # Using T_i(x) * T_j(y) for i, j in {0, 1, 2} plus one extra term T3(x)*T0(y)
    
    # i=0
    phi0 = T0_x * T0_y
    phi1 = T0_x * T1_y
    phi2 = T0_x * T2_y
    
    # i=1
    phi3 = T1_x * T0_y
    phi4 = T1_x * T1_y
    phi5 = T1_x * T2_y
    
    # i=2
    phi6 = T2_x * T0_y
    phi7 = T2_x * T1_y
    phi8 = T2_x * T2_y
    
    # Extra term to utilize 10th parameter
    phi9 = T3_x * T0_y
    
    # Linear combination
    return (params[0] * phi0 + params[1] * phi1 + params[2] * phi2 +
            params[3] * phi3 + params[4] * phi4 + params[5] * phi5 +
            params[6] * phi6 + params[7] * phi7 + params[8] * phi8 +
            params[9] * phi9)
