import numpy as np
import time

# Global start time for deadline guard
_start_time = None

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression with 10 parameters using scaled Chebyshev polynomials
    and bilinear interactions.
    
    Basis:
    1. Constant
    2-4. Linear (T1) for x, y, z
    5-7. Quadratic (T2) for x, y, z
    8-10. Bilinear interactions (T1x*T1y, T1x*T1z, T1y*T1z)
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: 150s limit. Safety margin 10s.
    if time.time() - _start_time > 140:
        # Return a safe prediction (e.g., mean of inputs or zeros) if timeout
        # In practice, this vectorized op is ~microseconds, so this won't trigger.
        return np.zeros_like(x)

    p = params
    
    # Map inputs [0.5, 2] to [-1, 1]
    # center = 1.25, half_range = 0.75
    tx = (x - 1.25) / 0.75
    ty = (y - 1.25) / 0.75
    tz = (z - 1.25) / 0.75
    
    # Chebyshev polynomials
    T0 = np.ones_like(tx)
    
    T1_x = tx
    T1_y = ty
    T1_z = tz
    
    T2_x = 2 * tx**2 - 1
    T2_y = 2 * ty**2 - 1
    T2_z = 2 * tz**2 - 1
    
    # Interactions
    I_xy = T1_x * T1_y
    I_xz = T1_x * T1_z
    I_yz = T1_y * T1_z
    
    return (p[0] * T0 +
            p[1] * T1_x +
            p[2] * T1_y +
            p[3] * T1_z +
            p[4] * T2_x +
            p[5] * T2_y +
            p[6] * T2_z +
            p[7] * I_xy +
            p[8] * I_xz +
            p[9] * I_yz)
