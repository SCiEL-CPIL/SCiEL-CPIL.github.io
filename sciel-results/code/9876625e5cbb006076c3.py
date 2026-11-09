import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Compute basis functions
    f1 = np.ones_like(x)
    f2 = x
    f3 = y
    f4 = x**2
    f5 = y**2
    f6 = x * y
    f7 = x**3
    f8 = y**3
    f9 = x**2 * y
    f10 = x * y**2
    
    # Linear combination
    pred = (params[0] * f1 + 
            params[1] * f2 + 
            params[2] * f3 + 
            params[3] * f4 + 
            params[4] * f5 + 
            params[5] * f6 + 
            params[6] * f7 + 
            params[7] * f8 + 
            params[8] * f9 + 
            params[9] * f10)
            
    return pred
