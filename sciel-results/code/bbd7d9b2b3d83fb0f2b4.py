"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    Uses a 3rd-order polynomial in scaled x and v to capture nonlinear dynamics.
    Scaling inputs helps the BFGS optimizer converge by keeping parameters O(1).
    
    params: [bias, x_s, v_s, x_s^2, v_s^2, x_s*v_s, x_s^3, v_s^3, x_s^2*v_s, x_s*v_s^2]
    """
    # Scale inputs to approximately [-1, 1] range to stabilize optimization
    # Training ranges: x ~ [-0.57, 0.53], v ~ [-0.18, 0.28]
    x_s = x / 0.5
    v_s = v / 0.25
    
    x2 = x_s * x_s
    v2 = v_s * v_s
    x3 = x2 * x_s
    v3 = v2 * v_s
    
    return (params[0] + 
            params[1] * x_s + 
            params[2] * v_s + 
            params[3] * x2 + 
            params[4] * v2 + 
            params[5] * x_s * v_s + 
            params[6] * x3 + 
            params[7] * v3 + 
            params[8] * x2 * v_s + 
            params[9] * x_s * v2)
# EVOLVE-BLOCK-END
