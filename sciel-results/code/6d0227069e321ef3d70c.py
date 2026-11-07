import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a sum of 3 Gaussian-like exponential terms.
    
    Model:
    f(x, y, z) = c0 + sum_{i=1}^{3} c_i * exp(-w_i * || [x, y, z] - center_i ||^2)
    
    Parameters (length 10):
    - params[0]: c0 (constant offset)
    - params[1:4]: center 1 (cx1, cy1, cz1)
    - params[4:7]: center 2 (cx2, cy2, cz2)
    - params[7:10]: center 3 (cx3, cy3, cz3)
    """
    c0 = params[0]
    
    cx1, cy1, cz1 = params[1], params[2], params[3]
    cx2, cy2, cz2 = params[4], params[5], params[6]
    cx3, cy3, cz3 = params[7], params[8], params[9]
    
    d1 = (x - cx1)**2 + (y - cy1)**2 + (z - cz1)**2
    d2 = (x - cx2)**2 + (y - cy2)**2 + (z - cz2)**2
    d3 = (x - cx3)**2 + (y - cy3)**2 + (z - cz3)**2
    
    result = c0 + np.exp(-d1) + np.exp(-d2) + np.exp(-d3)
    return result
