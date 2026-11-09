"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT_MARGIN = 148.0  # seconds, keep a safety margin before 150s kill

def _time_left():
    return _start_time + _TIMEOUT_MARGIN - time.time()

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters that tries to recover an unknown function.
    Strategy: Use a rich basis of common functional forms and let BFGS fit the 10 params.
    We choose a basis that covers linear, polynomial, trig, exp, rational, and mixed terms.
    
    The key insight: since we don't know the target function, we use a flexible 
    combination of common mathematical functions. With 10 parameters, we can afford
    a moderate basis. We pick basis functions that are smooth and well-behaved on [-1,1]^2.
    
    Basis functions chosen (10 total):
    1. 1 (constant)
    2. x
    3. y
    4. x^2
    5. y^2
    6. x*y
    7. x^3
    8. y^3
    9. sin(x)
    10. sin(y)
    
    This gives good coverage of low-order polynomial and simple trig behavior.
    """
    # Ensure inputs are arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Build basis matrix
    # Each column is a basis function evaluated at all points
    ones = np.ones_like(x)
    x2 = x * x
    y2 = y * y
    xy = x * y
    x3 = x2 * x
    y3 = y2 * y
    sinx = np.sin(x)
    siny = np.sin(y)
    
    # Stack basis functions: 10 columns
    B = np.column_stack([
        ones,   # 0
        x,      # 1
        y,      # 2
        x2,     # 3
        y2,     # 4
        xy,     # 5
        x3,     # 6
        y3,     # 7
        sinx,   # 8
        siny,   # 9
    ])
    
    # Prediction: B @ params
    pred = B @ params
    return pred

# EVOLVE-BLOCK-END
