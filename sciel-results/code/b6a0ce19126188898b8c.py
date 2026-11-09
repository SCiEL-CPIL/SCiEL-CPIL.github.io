import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Chebyshev polynomial basis 
    on scaled inputs.
    """
    # Scale x and v to approximately [-1, 1] based on training ranges
    # x: [-0.5728, 0.5275] -> mid -0.02265, half-range 0.55015
    # v: [-0.1754, 0.2821] -> mid 0.05335, half-range 0.22875
    
    x_mid = -0.02265
    x_scale = 0.55015
    v_mid = 0.05335
    v_scale = 0.22875
    
    xn = (x - x_mid) / x_scale
    vn = (v - v_mid) / v_scale
    
    # Clip to [-1, 1] to avoid numerical issues if test data slightly out of range
    # though task says in-distribution test split
    xn = np.clip(xn, -1.0, 1.0)
    vn = np.clip(vn, -1.0, 1.0)
    
    # Chebyshev polynomials T0..T3
    T0_x = np.ones_like(xn)
    T1_x = xn
    T2_x = 2.0 * xn**2 - 1.0
    T3_x = 4.0 * xn**3 - 3.0 * xn
    
    T0_v = np.ones_like(vn)
    T1_v = vn
    T2_v = 2.0 * vn**2 - 1.0
    T3_v = 4.0 * vn**3 - 3.0 * vn
    
    # Build 10 basis terms
    # Order: 
    # 0: T0(x)T0(v)
    # 1: T1(x)T0(v)
    # 2: T0(x)T1(v)
    # 3: T2(x)T0(v)
    # 4: T1(x)T1(v)
    # 5: T0(x)T2(v)
    # 6: T3(x)T0(v)
    # 7: T2(x)T1(v)
    # 8: T1(x)T2(v)
    # 9: T0(x)T3(v)
    
    b0 = T0_x * T0_v
    b1 = T1_x * T0_v
    b2 = T0_x * T1_v
    b3 = T2_x * T0_v
    b4 = T1_x * T1_v
    b5 = T0_x * T2_v
    b6 = T3_x * T0_v
    b7 = T2_x * T1_v
    b8 = T1_x * T2_v
    b9 = T0_x * T3_v
    
    return (params[0] * b0 + 
            params[1] * b1 + 
            params[2] * b2 + 
            params[3] * b3 + 
            params[4] * b4 + 
            params[5] * b5 + 
            params[6] * b6 + 
            params[7] * b7 + 
            params[8] * b8 + 
            params[9] * b9)
