"""Symbolic regression: nonlinear oscillator acceleration from position and velocity."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form model for acceleration of a nonlinear damped oscillator.
    
    Uses a physically-motivated basis with rescaled inputs for numerical stability:
    - Position terms: x, x^2, x^3 (restoring forces, Duffing-type)
    - Velocity terms: v, v^2, v^3 (damping, van der Pol-type)
    - Cross terms: x*v, x^2*v, x*v^2 (coupling)
    - Constant offset
    
    Inputs are rescaled to unit magnitude for better conditioning of the Hessian.
    """
    # Rescale inputs to improve condition number
    # x range: [-0.5728, 0.5275], center ~ -0.0227, scale ~ 0.55
    # v range: [-0.1754, 0.2821], center ~ 0.0534, scale ~ 0.23
    xc = (x + 0.0227) / 0.55
    vc = (v - 0.0534) / 0.23
    
    # Build basis functions
    x1 = xc
    x2 = xc ** 2
    x3 = xc ** 3
    v1 = vc
    v2 = vc ** 2
    v3 = vc ** 3
    xv = xc * vc
    x2v = xc ** 2 * vc
    xv2 = xc * vc ** 2
    
    # Linear combination with 10 parameters
    return (params[0] * x1 +
            params[1] * x2 +
            params[2] * x3 +
            params[3] * v1 +
            params[4] * v2 +
            params[5] * v3 +
            params[6] * xv +
            params[7] * x2v +
            params[8] * xv2 +
            params[9])
# EVOLVE-BLOCK-END
