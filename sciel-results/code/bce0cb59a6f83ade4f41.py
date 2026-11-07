import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0

def _elapsed_ok():
    return (time.time() - _START_TIME) < _DEADLINE

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Predicts y using a mixed basis of polynomials, trigonometric, exponential, and hyperbolic functions.
    x: input array
    params: array of 10 coefficients
    Returns: predictions array same shape as x
    """
    if not _elapsed_ok():
        return np.zeros_like(x)

    # Compute basis functions
    # 1. Constant
    b0 = np.ones_like(x)
    # 2. Linear
    b1 = x
    # 3. Quadratic
    b2 = x * x
    # 4. Cubic
    b3 = b2 * x
    # 5. Quartic
    b4 = b2 * b2
    
    # 6. Sin
    b5 = np.sin(x)
    # 7. Cos
    b6 = np.cos(x)
    
    # 8. Exp(x/3) - scaled to keep values reasonable
    b7 = np.exp(x / 3.0)
    # 9. Exp(-x/3)
    b8 = np.exp(-x / 3.0)
    
    # 10. Tanh
    b9 = np.tanh(x)
    
    # Combine
    y = (params[0] * b0 + 
         params[1] * b1 + 
         params[2] * b2 + 
         params[3] * b3 + 
         params[4] * b4 + 
         params[5] * b5 + 
         params[6] * b6 + 
         params[7] * b7 + 
         params[8] * b8 + 
         params[9] * b9)
         
    return y
