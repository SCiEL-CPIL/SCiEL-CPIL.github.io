import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Polynomial interaction model.
    params: [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9]
    y = c0 + c1*x + c2*v + c3*x^2 + c4*v^2 + c5*x*v 
        + c6*t_norm + c7*t_norm*x + c8*t_norm*v + c9*t_norm^2
    where t_norm = (t - 40) / 10
    """
    c0 = params[0]
    c1 = params[1]
    c2 = params[2]
    c3 = params[3]
    c4 = params[4]
    c5 = params[5]
    c6 = params[6]
    c7 = params[7]
    c8 = params[8]
    c9 = params[9]
    
    # Normalize time to center around 0 and scale to ~[-1, 1]
    t_norm = (t - 40.0) / 10.0
    
    result = (
        c0 
        + c1 * x 
        + c2 * v 
        + c3 * (x ** 2) 
        + c4 * (v ** 2) 
        + c5 * (x * v) 
        + c6 * t_norm 
        + c7 * (t_norm * x) 
        + c8 * (t_norm * v) 
        + c9 * (t_norm ** 2)
    )
    
    return result
