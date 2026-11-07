import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: polynomial + trigonometric + rational terms.
    Designed so that with params=all_ones, the output is bounded and smooth,
    helping BFGS converge from its starting point.
    
    Basis: 1, x, y, x*y, x^2, y^2, sin(x), sin(y), cos(x), cos(y)
    """
    xy = x * y
    x2 = x * x
    y2 = y * y

    pred = (params[0] 
            + params[1] * x 
            + params[2] * y 
            + params[3] * xy 
            + params[4] * x2 
            + params[5] * y2 
            + params[6] * np.sin(x) 
            + params[7] * np.sin(y) 
            + params[8] * np.cos(x) 
            + params[9] * np.cos(y))
            
    return pred
