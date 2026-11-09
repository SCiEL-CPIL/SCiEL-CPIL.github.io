"""Symbolic regression: nonlinear oscillator equation."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Damped nonlinear oscillator with time-dependent driving force.
    
    params[0]: linear spring constant
    params[1]: cubic spring constant (Duffing term)
    params[2]: linear damping
    params[3]: cubic damping
    params[4]: quadratic spring (x^2)
    params[5]: quadratic damping (v^2)
    params[6]: cross-term (x*v)
    params[7]: drive amplitude
    params[8]: drive frequency
    params[9]: constant offset
    """
    # Scale factors to keep terms O(1) at initialization
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    
    # Drive term: params[7] * sin(params[8] * t)
    drive = params[7] * np.sin(params[8] * t)
    
    result = (
        params[0] * x +
        params[1] * x3 +
        params[2] * v +
        params[3] * v3 +
        params[4] * x2 +
        params[5] * v2 +
        params[6] * xv +
        drive +
        params[9]
    )
    
    return result
# EVOLVE-BLOCK-END
