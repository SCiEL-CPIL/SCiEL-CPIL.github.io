import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator model with Duffing, Van der Pol, and sinusoidal drive.
    """
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = params
    
    # Drive terms
    # Using sin and cos with shared frequency c8
    drive_sin = np.sin(c8 * t)
    drive_cos = np.cos(c8 * t)
    
    # State terms
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    x2v = x2 * v
    xv = x * v
    
    result = (c0 
              + c1 * x 
              + c2 * x3 
              + c3 * v 
              + c4 * x2v 
              + c5 * drive_sin 
              + c6 * drive_cos 
              + c7 * xv 
              + c9 * v2)
              
    return result
