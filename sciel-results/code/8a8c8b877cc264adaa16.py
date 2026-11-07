import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation for symbolic regression with 2 inputs x, y.
    Uses a hybrid basis of polynomial and trigonometric terms.
    The 10 parameters correspond to the weights of these basis functions.
    
    Basis functions:
    0: 1 (bias)
    1: x
    2: y
    3: x^2
    4: y^2
    5: xy
    6: sin(pi * x)
    7: sin(pi * y)
    8: cos(pi * x)
    9: cos(pi * y)
    """
    # Deadline guard: start timer
    t0 = time.time()
    DEADLINE = 145.0  # 5s margin before 150s kill
    
    # Ensure inputs are float64 arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Check for non-finite inputs (safety)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(p))):
        return np.zeros_like(x)

    # Compute basis functions
    # Polynomials
    b1 = np.ones_like(x)
    b2 = x
    b3 = y
    b4 = x * x
    b5 = y * y
    b6 = x * y
    
    # Trigonometric (scaled to [-1, 1] domain, pi*x covers [-pi, pi])
    # Using pi scaling ensures full period coverage over the domain
    b7 = np.sin(np.pi * x)
    b8 = np.sin(np.pi * y)
    b9 = np.cos(np.pi * x)
    b10 = np.cos(np.pi * y)
    
    # Combine linearly
    # p[0]..p[9] map to b1..b10
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
        
    # Deadline check (unlikely to trigger for linear basis, but required)
    if time.time() - t0 > DEADLINE:
        # Return current prediction, which is already computed
        pass
        
    return pred
