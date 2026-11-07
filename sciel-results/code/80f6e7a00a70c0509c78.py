"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator.
    Uses normalized inputs to improve numerical conditioning for the optimizer.
    Model: Full 3rd order polynomial in normalized x and v.
    """
    # Check deadline (though not strictly necessary for a single eval, good practice)
    # if time.time() > _DEADLINE:
    #     return np.zeros_like(x)

    p = params
    
    # Normalize inputs to O(1) scale to help BFGS converge from all-ones init
    # Using approximate max abs values from training range
    x_scale = 0.6 
    v_scale = 0.3
    
    xn = x / x_scale
    vn = v / v_scale
    
    # Precompute powers
    xn2 = xn * xn
    xn3 = xn2 * xn
    vn2 = vn * vn
    vn3 = vn2 * vn
    
    # Terms
    # Order: 1, xn, vn, xn^2, xn*vn, vn^2, xn^3, xn^2*vn, xn*vn^2, vn^3
    return (
        p[0] +
        p[1] * xn +
        p[2] * vn +
        p[3] * xn2 +
        p[4] * xn * vn +
        p[5] * vn2 +
        p[6] * xn3 +
        p[7] * xn2 * vn +
        p[8] * xn * vn2 +
        p[9] * vn3
    )
# EVOLVE-BLOCK-END
