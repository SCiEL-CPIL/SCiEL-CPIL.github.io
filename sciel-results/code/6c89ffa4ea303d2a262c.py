import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    c_x = params[0]
    c_x3 = params[1]
    c_x5 = params[2]
    c_v = params[3]
    c_v2 = params[4]
    c_v3 = params[5]
    c_xv = params[6]
    c_x2v = params[7]
    c_x2 = params[8]
    c_bias = params[9]
    
    # Restoring forces (position-dependent)
    restoring = c_x * x + c_x2 * (x ** 2) + c_x3 * (x ** 3) + c_x5 * (x ** 5)
    
    # Damping (velocity-dependent)
    damping = c_v * v + c_v2 * (v ** 2) + c_v3 * (v ** 3)
    
    # Cross coupling
    coupling = c_xv * (x * v) + c_x2v * (x ** 2 * v)
    
    result = -restoring - damping - coupling + c_bias
    
    return result
