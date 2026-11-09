# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a scaled monomial basis.
    
    Structural changes from previous iterations:
    1. Input scaling: Maps x and v to approximately [-1, 1] to improve 
       numerical conditioning of the polynomial terms.
    2. Symmetry Enforcement: Uses only odd monomials (total degree odd), 
       which is physically consistent for an unforced oscillator's acceleration.
    3. Basis Selection: Includes up to 5th order terms to capture nonlinearity 
       without the ill-conditioning of high powers on unscaled inputs.
    
    The model is:
        a = sum( p_i * basis_i(u, w) )
    where u = x / x_scale, w = v / v_scale.
    """
    # Deadline guard (safety margin, though equation is fast)
    start_time = time.time()
    deadline = start_time + 1.0  # 1 second limit for this function call
    
    # Scaling factors based on observed ranges
    # x in [-0.5728, 0.5275] -> scale 0.6
    # v in [-0.1754, 0.2821] -> scale 0.3
    x_scale = 0.6
    v_scale = 0.3
    
    u = x / x_scale
    w = v / v_scale
    
    # Precompute powers of u and w
    u2 = u * u
    u3 = u2 * u
    u4 = u2 * u2
    u5 = u4 * u
    
    w2 = w * w
    w3 = w2 * w
    w4 = w2 * w2
    w5 = w4 * w
    
    # Basis functions (odd monomials)
    # Degree 1
    b0 = u
    b1 = w
    
    # Degree 3
    b2 = u3
    b3 = u2 * w
    b4 = u * w2
    b5 = w3
    
    # Degree 5
    b6 = u5
    b7 = u4 * w
    b8 = u3 * w2
    b9 = u * w4
    
    # Combine with parameters
    p = params
    
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
         
    # Check deadline (unlikely to trigger, but required by directive)
    if time.time() > deadline:
        # Return a simple fallback if we somehow timed out (should not happen)
        return p[0] * x + p[1] * v
        
    return y
# EVOLVE-BLOCK-END
