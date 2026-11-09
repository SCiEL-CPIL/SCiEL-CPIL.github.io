import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a cubic polynomial basis
    in normalized position and velocity.
    """
    # Normalize inputs to O(1) range to improve conditioning for BFGS
    # x range approx [-0.6, 0.6], v range approx [-0.3, 0.3]
    # Using 0.6 and 0.3 as scales
    x_n = x / 0.6
    v_n = v / 0.3
    
    # Precompute powers
    x2 = x_n * x_n
    v2 = v_n * v_n
    x3 = x2 * x_n
    v3 = v2 * v_n
    
    # Construct basis terms
    # Order matches params[0] to params[9]
    t0 = np.ones_like(x)
    t1 = x_n
    t2 = v_n
    t3 = x2
    t4 = x_n * v_n
    t5 = v2
    t6 = x3
    t7 = x2 * v_n
    t8 = x_n * v2
    t9 = v3
    
    return (params[0] * t0 +
            params[1] * t1 +
            params[2] * t2 +
            params[3] * t3 +
            params[4] * t4 +
            params[5] * t5 +
            params[6] * t6 +
            params[7] * t7 +
            params[8] * t8 +
            params[9] * t9)
