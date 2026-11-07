import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation for symbolic regression with 2 inputs x, y.
    Combines low-order polynomial terms with bounded tanh non-linearities.
    The 10 parameters correspond to the weights of these basis functions.
    
    Basis functions:
    0: 1 (bias)
    1: x
    2: y
    3: x * y
    4: x * x
    5: y * y
    6: tanh(2.0 * x)
    7: tanh(2.0 * y)
    8: tanh(2.0 * (x + y))
    9: tanh(2.0 * (x - y))
    """
    # Deadline guard
    t0 = time.time()
    DEADLINE = 140.0  # 10s margin before 150s kill
    
    # Ensure inputs are float64 arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # 1. Polynomial Basis (Low order for stability and global structure)
    b1 = np.ones_like(x)       # Bias
    b2 = x                     # Linear x
    b3 = y                     # Linear y
    b4 = x * y                 # Interaction
    b5 = x * x                 # Quadratic x
    b6 = y * y                 # Quadratic y
    
    # 2. Bounded tanh non-linearities (well-conditioned for BFGS)
    b7 = np.tanh(2.0 * x)
    b8 = np.tanh(2.0 * y)
    b9 = np.tanh(2.0 * (x + y))
    b10 = np.tanh(2.0 * (x - y))
    
    # Combine linearly
    # The order of parameters p[0]..p[9] maps to b1..b10
    pred = (p[0] * b1 + 
            p[1] * b2 + 
            p[2] * b3 + 
            p[3] * b4 + 
            p[4] * b5 + 
            p[5] * b6 + 
            p[6] * b7 + 
            p[7] * b8 + 
            p[8] * b9 + 
            p[9] * b10)
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, 0.0)
    
    # Deadline guard check
    if time.time() - t0 > DEADLINE:
        return pred
    
    return pred
