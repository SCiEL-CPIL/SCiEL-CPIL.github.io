import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.ravel()
        
    # Basis functions
    b1 = np.ones_like(x)
    b2 = x
    b3 = x**2
    b4 = x**3
    b5 = x**4
    b6 = np.sin(x)
    b7 = np.cos(x)
    b8 = np.tanh(x)
    b9 = np.exp(-x**2)
    b10 = x * np.exp(-x**2)
    
    # Combine
    y = (params[0] * b1 + 
         params[1] * b2 + 
         params[2] * b3 + 
         params[3] * b4 + 
         params[4] * b5 + 
         params[5] * b6 + 
         params[6] * b7 + 
         params[7] * b8 + 
         params[8] * b9 + 
         params[9] * b10)
         
    # Safety: replace non-finite values with 0 to avoid crashing the evaluator
    y = np.where(np.isfinite(y), y, 0.0)
    
    return y
