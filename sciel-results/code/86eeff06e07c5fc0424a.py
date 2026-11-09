import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a hybrid basis:
    combines low-order monomials, Chebyshev polynomials, and trigonometric
    terms to capture both polynomial trends and oscillatory structure.
    
    Basis (10 terms on normalized inputs xn, vn):
    0: 1 (bias)
    1: xn
    2: vn
    3: xn^2
    4: vn^2
    5: xn * vn
    6: T2(xn) * T1(vn)  = (2*xn^2 - 1) * vn
    7: sin(xn)
    8: cos(xn) * vn
    9: sin(vn)
    """
    # Normalization constants based on training ranges
    # x: [-0.5728, 0.5275] -> mid -0.02265, half-range 0.55015
    # v: [-0.1754, 0.2821] -> mid 0.05335, half-range 0.22875
    
    x_mid = -0.02265
    x_scale = 0.55015
    v_mid = 0.05335
    v_scale = 0.22875
    
    xn = (x - x_mid) / x_scale
    vn = (v - v_mid) / v_scale
    
    # Clip to avoid numerical issues if slightly out of range
    xn = np.clip(xn, -1.5, 1.5)
    vn = np.clip(vn, -1.5, 1.5)
    
    p = params
    
    # Build basis terms
    # Low-order monomials
    t0 = np.ones_like(xn)
    t1 = xn
    t2 = vn
    t3 = xn * xn
    t4 = vn * vn
    t5 = xn * vn
    
    # Chebyshev-inspired interaction term
    T2_xn = 2.0 * xn * xn - 1.0
    t6 = T2_xn * vn
    
    # Trigonometric terms to capture oscillatory structure
    t7 = np.sin(xn)
    t8 = np.cos(xn) * vn
    t9 = np.sin(vn)
    
    # Combine
    return (p[0] * t0 + 
            p[1] * t1 + 
            p[2] * t2 + 
            p[3] * t3 + 
            p[4] * t4 + 
            p[5] * t5 + 
            p[6] * t6 + 
            p[7] * t7 + 
            p[8] * t8 + 
            p[9] * t9)
