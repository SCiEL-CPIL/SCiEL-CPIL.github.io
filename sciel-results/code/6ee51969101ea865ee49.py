import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model with fitted driving frequency.
    
    y = b + c1*x + c2*x^2 + c3*x^3 + c4*v + c5*x*v + c6*x^2*v + A*sin(w*t) + B*cos(w*t)
    
    params:
    0: b (bias)
    1: c1 (linear spring)
    2: c2 (quadratic spring)
    3: c3 (cubic spring)
    4: c4 (linear damping)
    5: c5 (cross term x*v)
    6: c6 (Van der Pol term x^2*v)
    7: A (driving amplitude sin)
    8: B (driving amplitude cos)
    9: w (driving frequency)
    """
    b = params[0]
    c1 = params[1]
    c2 = params[2]
    c3 = params[3]
    c4 = params[4]
    c5 = params[5]
    c6 = params[6]
    A = params[7]
    B = params[8]
    w = params[9]
    
    x2 = x * x
    x3 = x2 * x
    xv = x * v
    x2v = x2 * v
    
    wt = w * t
    sin_wt = np.sin(wt)
    cos_wt = np.cos(wt)
    
    y = (b + 
         c1 * x + 
         c2 * x2 + 
         c3 * x3 + 
         c4 * v + 
         c5 * xv + 
         c6 * x2v + 
         A * sin_wt + 
         B * cos_wt)
         
    return y
