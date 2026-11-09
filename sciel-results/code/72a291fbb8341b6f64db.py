import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a quadratic response surface on centered inputs.
    This structure captures curvature (saturation, peaks) without the initialization
    difficulties of Gaussian/Haldane terms, as centers are fixed.
    """
    # Hardcoded centers to aid optimization from params=ones
    b_c = b - 3.0
    s_c = s - 10.0
    t_c = temp - 30.0
    p_c = pH - 7.0
    
    p = params
    
    # Quadratic terms
    b2 = b_c * b_c
    s2 = s_c * s_c
    t2 = t_c * t_c
    p2 = p_c * p_c
    
    # Interactions (selecting key ones to fit in 10 params)
    # p0: intercept
    # p1-p4: linear
    # p5-p8: quadratic
    # p9: temp-pH interaction (common in growth)
    
    return (p[0] + 
            p[1] * b_c + 
            p[2] * s_c + 
            p[3] * t_c + 
            p[4] * p_c + 
            p[5] * b2 + 
            p[6] * s2 + 
            p[7] * t2 + 
            p[8] * p2 + 
            p[9] * t_c * p_c)
