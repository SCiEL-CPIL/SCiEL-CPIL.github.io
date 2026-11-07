"""Symbolic regression for a nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # Safety margin before 120s limit

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator.
    
    Uses a physics-informed bounded nonlinear basis (sin, tanh, cubic) 
    to capture periodic restoring and saturation effects.
    
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

    # Basis functions:
    # 0: 1 (Bias)
    # 1: x (Linear restoring force)
    # 2: v (Linear damping)
    # 3: x^3 (Cubic restoring force - Duffing)
    # 4: sin(x) (Bounded/Periodic restoring force)
    # 5: tanh(x) (Saturation restoring force)
    # 6: v*x (Cross term / Parametric)
    # 7: v^3 (Cubic damping)
    # 8: v*sin(x) (Nonlinear damping)
    # 9: v*tanh(x) (Nonlinear damping)
    
    x2 = x * x
    x3 = x2 * x
    
    sin_x = np.sin(x)
    tanh_x = np.tanh(x)
    
    v2 = v * v
    v3 = v2 * v
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = v
    b3 = x3
    b4 = sin_x
    b5 = tanh_x
    b6 = v * x
    b7 = v3
    b8 = v * sin_x
    b9 = v * tanh_x
    
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
