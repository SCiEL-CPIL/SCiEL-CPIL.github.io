import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Chebyshev polynomial basis.
    
    The model is a linear combination of 10 Chebyshev basis functions:
    phi_0 = 1
    phi_1 = x
    phi_2 = y
    phi_3 = 2x^2 - 1
    phi_4 = 2y^2 - 1
    phi_5 = xy
    phi_6 = 4x^3 - 3x
    phi_7 = 4y^3 - 3y
    phi_8 = (2x^2 - 1)y
    phi_9 = x(2y^2 - 1)
    
    params: array of 10 weights [w0, ..., w9]
    """
    # Compute basis functions
    # T0(x) = 1
    # T1(x) = x
    # T2(x) = 2x^2 - 1
    # T3(x) = 4x^3 - 3x
    
    T1_x = x
    T2_x = 2 * x**2 - 1
    T3_x = 4 * x**3 - 3 * x
    
    T1_y = y
    T2_y = 2 * y**2 - 1
    T3_y = 4 * y**3 - 3 * y
    
    # Construct the 10 basis vectors
    # phi_0
    phi0 = np.ones_like(x)
    # phi_1
    phi1 = T1_x
    # phi_2
    phi2 = T1_y
    # phi_3
    phi3 = T2_x
    # phi_4
    phi4 = T2_y
    # phi_5
    phi5 = T1_x * T1_y
    # phi_6
    phi6 = T3_x
    # phi_7
    phi7 = T3_y
    # phi_8
    phi8 = T2_x * T1_y
    # phi_9
    phi9 = T1_x * T2_y
    
    # Stack into a design matrix of shape (N, 10)
    # Note: We can do this without creating a full matrix if we want to be memory efficient,
    # but N=200 is small, so (200, 10) is fine.
    # However, computing dot products directly is also fast.
    
    # Let's just compute the weighted sum directly to avoid matrix allocation overhead if any,
    # though numpy broadcasting handles this well.
    
    pred = (params[0] * phi0 + 
            params[1] * phi1 + 
            params[2] * phi2 + 
            params[3] * phi3 + 
            params[4] * phi4 + 
            params[5] * phi5 + 
            params[6] * phi6 + 
            params[7] * phi7 + 
            params[8] * phi8 + 
            params[9] * phi9)
            
    return pred
