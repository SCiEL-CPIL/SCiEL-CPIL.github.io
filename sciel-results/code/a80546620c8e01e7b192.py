import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a polynomial basis.
    Includes linear, quadratic, and cubic terms in x, quadratic in v,
    and interaction terms with time t.
    """
    p = params
    
    # Basis terms
    # 1. Bias
    term_0 = 1.0
    # 2. Linear x
    term_1 = x
    # 3. Linear v
    term_2 = v
    # 4. Quadratic x (softening/hardening spring start)
    term_3 = x * x
    # 5. Cubic x (Duffing term)
    term_4 = x * x * x
    # 6. Quadratic v (nonlinear damping)
    term_5 = v * v
    # 7. Interaction x*v
    term_6 = x * v
    # 8. Linear t (drift or linear drive approximation)
    term_7 = t
    # 9. Interaction x*t
    term_8 = x * t
    # 10. Interaction v*t
    term_9 = v * t
    
    # Combine
    pred = (p[0] * term_0 + 
            p[1] * term_1 + 
            p[2] * term_2 + 
            p[3] * term_3 + 
            p[4] * term_4 + 
            p[5] * term_5 + 
            p[6] * term_6 + 
            p[7] * term_7 + 
            p[8] * term_8 + 
            p[9] * term_9)
            
    return pred
