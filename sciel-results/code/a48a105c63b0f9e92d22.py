import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Autonomous nonlinear oscillator model.
    Uses a cubic polynomial basis in position (x) and velocity (v).
    Basis: 1, x, v, x^2, xv, v^2, x^3, x^2v, xv^2, v^3
    """
    p = params
    
    # Pre-compute powers
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    x2v = x2 * v
    xv2 = x * v2
    
    # Linear combination
    # Order:
    # 0: 1
    # 1: x
    # 2: v
    # 3: x^2
    # 4: xv
    # 5: v^2
    # 6: x^3
    # 7: x^2v
    # 8: xv^2
    # 9: v^3
    
    return (p[0] + 
            p[1] * x + 
            p[2] * v + 
            p[3] * x2 + 
            p[4] * xv + 
            p[5] * v2 + 
            p[6] * x3 + 
            p[7] * x2v + 
            p[8] * xv2 + 
            p[9] * v3)
