import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure x is a 1D array for consistent indexing, though broadcasting usually works
    x = np.asarray(x, dtype=float)
    
    # Calculate basis functions
    # p0: constant
    # p1: x
    # p2: x^2
    # p3: x^3
    # p4: sin(x)
    # p5: cos(x)
    # p6: sin(2x)
    # p7: cos(2x)
    # p8: sin(3x)
    # p9: cos(3x)
    
    # We can compute this efficiently by precomputing powers and trig functions
    x2 = x * x
    x3 = x2 * x
    
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sin_2x = np.sin(2 * x)
    cos_2x = np.cos(2 * x)
    sin_3x = np.sin(3 * x)
    cos_3x = np.cos(3 * x)
    
    # Combine with parameters
    y = (params[0] * 1.0 + 
         params[1] * x + 
         params[2] * x2 + 
         params[3] * x3 + 
         params[4] * sin_x + 
         params[5] * cos_x + 
         params[6] * sin_2x + 
         params[7] * cos_2x + 
         params[8] * sin_3x + 
         params[9] * cos_3x)
         
    return y
