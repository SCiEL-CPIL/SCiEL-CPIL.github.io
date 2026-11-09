import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a mixed polynomial and tanh basis.
    This approach incorporates saturation effects via tanh terms, which are common in 
    nonlinear oscillators, alongside standard polynomial terms for local linear/quadratic behavior.
    
    Terms:
    p0: x
    p1: v
    p2: x^2
    p3: v^2
    p4: x*v
    p5: tanh(x)
    p6: tanh(v)
    p7: tanh(x)*v
    p8: x*tanh(v)
    p9: tanh(x)*tanh(v)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Precompute tanh values for efficiency and reuse
    tanh_x = np.tanh(x)
    tanh_v = np.tanh(v)
    
    term_0 = p0 * x
    term_1 = p1 * v
    term_2 = p2 * x**2
    term_3 = p3 * v**2
    term_4 = p4 * x * v
    term_5 = p5 * tanh_x
    term_6 = p6 * tanh_v
    term_7 = p7 * tanh_x * v
    term_8 = p8 * x * tanh_v
    term_9 = p9 * tanh_x * tanh_v
    
    return term_0 + term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8 + term_9
