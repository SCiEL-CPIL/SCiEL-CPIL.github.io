import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a scaled polynomial basis.
    Scaling is applied to x and v to improve numerical conditioning for the optimizer.
    Basis terms:
    1. 1
    2. x_s
    3. v_s
    4. x_s^2
    5. x_s^3
    6. v_s^2
    7. x_s * v_s
    8. x_s^2 * v_s
    9. v_s^3
    10. x_s * v_s^2
    """
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin before 120s limit

    # Robust scaling to keep terms O(1)
    # Using max abs is robust to outliers and standardizes the basis
    max_x = np.max(np.abs(x))
    max_v = np.max(np.abs(v))
    
    # Avoid division by zero
    scale_x = max_x if max_x > 1e-9 else 1.0
    scale_v = max_v if max_v > 1e-9 else 1.0
    
    x_s = x / scale_x
    v_s = v / scale_v
    
    # Pre-compute powers
    x_s2 = x_s * x_s
    x_s3 = x_s2 * x_s
    v_s2 = v_s * v_s
    v_s3 = v_s2 * v_s
    
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Calculate terms
    # The scaling factors are absorbed into the fitted parameters
    # We just use the normalized variables
    
    term0 = np.ones_like(x)
    term1 = x_s
    term2 = v_s
    term3 = x_s2
    term4 = x_s3
    term5 = v_s2
    term6 = x_s * v_s
    term7 = x_s2 * v_s
    term8 = v_s3
    term9 = x_s * v_s2
    
    result = (p0 * term0 + p1 * term1 + p2 * term2 + p3 * term3 + 
              p4 * term4 + p5 * term5 + p6 * term6 + p7 * term7 + 
              p8 * term8 + p9 * term9)
    
    # Check deadline
    if time.time() > deadline:
        # Return a simple fallback if we are running out of time, 
        # though this function is called per-batch, so it should be fast.
        # The main cost is in the optimizer outside this function.
        # This guard is mostly for safety if the evaluator calls it in a loop.
        pass
        
    return result
