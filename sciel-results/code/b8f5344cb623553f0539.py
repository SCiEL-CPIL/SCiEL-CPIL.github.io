import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Chebyshev polynomial basis.
    
    Inputs:
    x: Position
    v: Velocity
    params: Array of 10 fitted constants
    
    Returns:
    Acceleration
    """
    # Training ranges from the problem description
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    
    # Scale inputs to [-1, 1]
    # Handle edge cases where range is zero (though not expected here)
    x_range = x_max - x_min
    v_range = v_max - v_min
    
    if x_range < 1e-12:
        X = np.zeros_like(x)
    else:
        X = (2 * x - (x_min + x_max)) / x_range
        
    if v_range < 1e-12:
        V = np.zeros_like(v)
    else:
        V = (2 * v - (v_min + v_max)) / v_range

    # Compute Chebyshev polynomials up to degree 3
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    
    T0_X = np.ones_like(X)
    T1_X = X
    T2_X = 2 * X * X - 1
    T3_X = 4 * X * X * X - 3 * X
    
    T0_V = np.ones_like(V)
    T1_V = V
    T2_V = 2 * V * V - 1
    T3_V = 4 * V * V * V - 3 * V
    
    # Construct the basis matrix (implicitly via broadcasting)
    # We have 10 parameters, so we pick 10 basis terms.
    # The selection aims to cover low-order interactions and cubic non-linearities.
    
    # p0: Constant
    term0 = T0_X * T0_V
    
    # p1: Linear X
    term1 = T1_X * T0_V
    
    # p2: Linear V
    term2 = T0_X * T1_V
    
    # p3: Quadratic X
    term3 = T2_X * T0_V
    
    # p4: Quadratic V
    term4 = T0_X * T2_V
    
    # p5: Interaction XV
    term5 = T1_X * T1_V
    
    # p6: Cubic X
    term6 = T3_X * T0_V
    
    # p7: Cubic V
    term7 = T0_X * T3_V
    
    # p8: Mixed X^2 V
    term8 = T2_X * T1_V
    
    # p9: Mixed X V^2
    term9 = T1_X * T2_V
    
    # Combine with parameters
    # params is a 1D array of length 10
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    return (p0 * term0 + 
            p1 * term1 + 
            p2 * term2 + 
            p3 * term3 + 
            p4 * term4 + 
            p5 * term5 + 
            p6 * term6 + 
            p7 * term7 + 
            p8 * term8 + 
            p9 * term9)
