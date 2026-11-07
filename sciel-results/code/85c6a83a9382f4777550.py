import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    10-parameter basis for symbolic regression on [-1,1] x [-1,1].
    Basis: 1, x, y, x^2, y^2, xy, tanh(x), tanh(y), x*tanh(y), y*tanh(x)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    tx = np.tanh(x)
    ty = np.tanh(y)
    
    return (p0 
            + p1 * x 
            + p2 * y 
            + p3 * x**2 
            + p4 * y**2 
            + p5 * x * y 
            + p6 * tx 
            + p7 * ty 
            + p8 * x * ty 
            + p9 * y * tx)
