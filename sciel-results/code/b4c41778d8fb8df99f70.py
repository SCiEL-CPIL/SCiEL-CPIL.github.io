"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Chebyshev polynomial basis.
    Chebyshev polynomials provide a well-conditioned basis for approximation
    over a bounded interval, which helps the BFGS optimizer converge reliably.
    The basis includes terms corresponding to linear and cubic stiffness/damping
    (Duffing-van der Pol dynamics) as well as cross terms.
    
    params: 
    0: Bias
    1: Linear x (T1_x)
    2: Linear v (T1_v)
    3: Quadratic x (T2_x)
    4: Quadratic v (T2_v)
    5: Cross x*v (T1_x * T1_v)
    6: Cubic x (T3_x)
    7: Cubic v (T3_v)
    8: x^2 v (T2_x * T1_v)
    9: x v^2 (T1_x * T2_v)
    """
    # Deadline guard (safety margin)
    start_time = time.time()
    
    # Convert inputs to numpy arrays if not already
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64).flatten()
    
    # Scale inputs to approximately [-1, 1] to stabilize optimization
    # Training ranges: x ~ [-0.57, 0.53], v ~ [-0.17, 0.28]
    # Using slightly larger scales to ensure inputs stay within [-1, 1]
    x_scale = 0.6
    v_scale = 0.3
    
    x_s = x / x_scale
    v_s = v / v_scale
    
    # Chebyshev Polynomials of the first kind
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # T3(t) = 4t^3 - 3t
    
    T1_x = x_s
    T2_x = 2.0 * x_s * x_s - 1.0
    T3_x = 4.0 * x_s * x_s * x_s - 3.0 * x_s
    
    T1_v = v_s
    T2_v = 2.0 * v_s * v_s - 1.0
    T3_v = 4.0 * v_s * v_s * v_s - 3.0 * v_s
    
    # Construct the linear combination
    # Note: T2_x and T3_x contain constant terms, so the "bias" param[0]
    # will interact with them. This is fine for a general approximation.
    
    pred = (params[0] + 
            params[1] * T1_x + 
            params[2] * T1_v + 
            params[3] * T2_x + 
            params[4] * T2_v + 
            params[5] * T1_x * T1_v + 
            params[6] * T3_x + 
            params[7] * T3_v + 
            params[8] * T2_x * T1_v + 
            params[9] * T1_x * T2_v)
            
    # Ensure output is finite
    if not np.all(np.isfinite(pred)):
        return np.zeros_like(x)
        
    # Check time (should be negligible, but good practice)
    if time.time() - start_time > 140:
         return pred

    return pred
# EVOLVE-BLOCK-END
