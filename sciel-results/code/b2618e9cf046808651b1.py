import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scaling factors based on input ranges
    x_scale = 0.21
    v_scale = 0.36
    
    xs = x / x_scale
    vs = v / v_scale
    
    # Chebyshev polynomials for position
    T0_x = 1.0
    T1_x = xs
    T2_x = 2.0 * xs**2 - 1.0
    T3_x = 4.0 * xs**3 - 3.0 * xs
    
    # Chebyshev polynomials for velocity
    T1_v = vs
    T2_v = 2.0 * vs**2 - 1.0
    T3_v = 4.0 * vs**3 - 3.0 * vs
    
    # Time terms
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Cross term
    cross = xs * vs
    
    # Combine with params
    # p0: sin(t)
    # p1: cos(t)
    # p2: constant (T0_x)
    # p3: T1_x
    # p4: T2_x
    # p5: T3_x
    # p6: T1_v
    # p7: T2_v
    # p8: T3_v
    # p9: cross term
    
    result = (params[0] * sin_t + 
              params[1] * cos_t + 
              params[2] * T0_x + 
              params[3] * T1_x + 
              params[4] * T2_x + 
              params[5] * T3_x + 
              params[6] * T1_v + 
              params[7] * T2_v + 
              params[8] * T3_v + 
              params[9] * cross)
              
    return result
