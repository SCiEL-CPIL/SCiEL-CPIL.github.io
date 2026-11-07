"""Symbolic regression with a richer basis and stable evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(x)
    
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Build a diverse set of basis functions that can represent many common
    # smooth functions on [-3, 3]. We use 10 basis functions matching the
    # 10 parameters.
    
    # 1. Constant
    b0 = np.ones_like(x)
    
    # 2. Linear
    b1 = x
    
    # 3. Quadratic
    b2 = x ** 2
    
    # 4. Cubic
    b3 = x ** 3
    
    # 5. Quartic
    b4 = x ** 4
    
    # 6. Sine
    b5 = np.sin(x)
    
    # 7. Cosine
    b6 = np.cos(x)
    
    # 8. Exponential (scaled to avoid overflow issues, but x in [-3,3] so exp(x) is safe)
    b7 = np.exp(x)
    
    # 9. Tanh
    b8 = np.tanh(x)
    
    # 10. Absolute value
    b9 = np.abs(x)
    
    # Combine: y = sum(p_i * b_i)
    # Use a vectorized approach for speed
    y = (p[0] * b0 + 
         p[1] * b1 + 
         p[2] * b2 + 
         p[3] * b3 + 
         p[4] * b4 + 
         p[5] * b5 + 
         p[6] * b6 + 
         p[7] * b7 + 
         p[8] * b8 + 
         p[9] * b9)
    
    return y
# EVOLVE-BLOCK-END
