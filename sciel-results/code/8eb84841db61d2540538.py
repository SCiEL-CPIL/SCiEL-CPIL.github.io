"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with time-dependent driving force."""
    # Features:
    # 0: bias
    # 1: x
    # 2: v
    # 3: x^2
    # 4: x^3
    # 5: v^2
    # 6: v^3
    # 7: x*v
    # 8: sin(t)
    # 9: cos(t)
    
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    
    return (params[0] + 
            params[1] * x + 
            params[2] * v + 
            params[3] * x2 + 
            params[4] * x3 + 
            params[5] * v2 + 
            params[6] * v3 + 
            params[7] * xv + 
            params[8] * np.sin(t) + 
            params[9] * np.cos(t))
# EVOLVE-BLOCK-END
