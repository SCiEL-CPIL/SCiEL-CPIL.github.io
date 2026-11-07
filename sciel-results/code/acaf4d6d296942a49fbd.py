"""
Symbolic regression with 1 input.
Uses a diverse mixed basis (polynomial, trigonometric, hyperbolic, exponential)
to provide BFGS with a stable and expressive manifold for fitting the hidden function.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: The evaluator kills after 150s.
# We set a conservative limit to ensure we return within time.
# However, this function is O(N) and very fast, so the guard is mostly precautionary.
_MAX_TIME = 140.0
_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Computes a linear combination of 10 basis functions.
    Basis:
    0: 1
    1: x
    2: x^2
    3: x^3
    4: x^4
    5: sin(x)
    6: cos(x)
    7: exp(-x^2)
    8: tanh(x)
    9: x * sin(x)  # Adds some non-monotonic curvature without high frequency
    """
    # Check time limit
    if time.time() - _START_TIME > _MAX_TIME:
        # Return a safe default if we are about to time out
        return np.zeros_like(x)

    x = np.asarray(x, dtype=np.float64)
    
    # Compute basis functions
    # Vectorized operations for speed
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = x * x
    b3 = b2 * x
    b4 = b2 * b2
    
    b5 = np.sin(x)
    b6 = np.cos(x)
    
    # exp(-x^2) is stable and bounded
    b7 = np.exp(-b2)
    
    # tanh(x) is stable and bounded
    b8 = np.tanh(x)
    
    # x*sin(x) provides a different curvature profile
    b9 = x * b5
    
    # Linear combination
    # params is length 10
    # p0*b0 + p1*b1 + ... + p9*b9
    
    pred = (params[0] * b0 + 
            params[1] * b1 + 
            params[2] * b2 + 
            params[3] * b3 + 
            params[4] * b4 + 
            params[5] * b5 + 
            params[6] * b6 + 
            params[7] * b7 + 
            params[8] * b8 + 
            params[9] * b9)
            
    return pred
# EVOLVE-BLOCK-END
