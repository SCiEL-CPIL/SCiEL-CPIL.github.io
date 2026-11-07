"""Symbolic regression for a nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator.
    
    Uses a physics-informed basis with scaled inputs to [-1, 1].
    The basis includes linear, quadratic, and cross terms that are
    likely relevant for oscillator dynamics (spring, damping, nonlinearity).
    
    Inputs:
        x: position
        v: velocity
        params: 10 fitted constants
    
    Returns:
        Acceleration prediction (one per row)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros_like(x)

    p = params

    # Training ranges from task description
    x_min, x_max = -0.5728, 0.5275
    v_min, v_max = -0.1754, 0.2821
    
    # Scale inputs to [-1, 1]
    x_range = x_max - x_min
    v_range = v_max - v_min
    
    if x_range < 1e-9:
        u = np.zeros_like(x)
    else:
        u = 2.0 * (x - x_min) / x_range - 1.0
        
    if v_range < 1e-9:
        w = np.zeros_like(v)
    else:
        w = 2.0 * (v - v_min) / v_range - 1.0

    # Basis functions:
    # 0: constant
    # 1: u (position)
    # 2: w (velocity)
    # 3: u^2 (quadratic position)
    # 4: w^2 (quadratic velocity)
    # 5: u*w (cross term)
    # 6: u^3 (cubic position)
    # 7: w^3 (cubic velocity)
    # 8: u^2*w (nonlinear coupling)
    # 9: u*w^2 (nonlinear coupling)
    
    u2 = u * u
    u3 = u2 * u
    w2 = w * w
    w3 = w2 * w
    
    b0 = np.ones_like(u)
    b1 = u
    b2 = w
    b3 = u2
    b4 = w2
    b5 = u * w
    b6 = u3
    b7 = w3
    b8 = u2 * w
    b9 = u * w2
    
    return (
        p[0] * b0 +
        p[1] * b1 +
        p[2] * b2 +
        p[3] * b3 +
        p[4] * b4 +
        p[5] * b5 +
        p[6] * b6 +
        p[7] * b7 +
        p[8] * b8 +
        p[9] * b9
    )
# EVOLVE-BLOCK-END
