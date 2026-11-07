"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration model with 10 fitted parameters.
    
    Features chosen to capture common nonlinear oscillator dynamics:
    - Linear terms: constant, t, x, v
    - Nonlinear position: x^2, x^3 (Duffing-like)
    - Nonlinear velocity: v^2, v^3 (drag-like)
    - Cross terms: x*v, x^2*v
    - Periodic forcing: sin(t), cos(t)
    
    We select a subset of 10 features from a richer basis to span
    common nonlinear oscillator behaviors.
    """
    # Check deadline
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback to simple linear model
        return params[0] + params[1] * t + params[2] * x + params[3] * v
    
    # Compute features
    x2 = x * x
    x3 = x2 * x
    v2 = v * v
    v3 = v2 * v
    xv = x * v
    x2v = x2 * v
    sint = np.sin(t)
    cost = np.cos(t)
    t2 = t * t
    t3 = t2 * t
    exp_neg_t = np.exp(-np.abs(t) * 0.1)  # bounded decay
    tanh_x = np.tanh(x)
    tanh_v = np.tanh(v)
    
    # 10 features: constant, t, x, v, x^2, x^3, v^2, x*v, sin(t), cos(t)
    # This is a standard nonlinear oscillator basis
    return (
        params[0]
        + params[1] * t
        + params[2] * x
        + params[3] * v
        + params[4] * x2
        + params[5] * x3
        + params[6] * v2
        + params[7] * xv
        + params[8] * sint
        + params[9] * cost
    )
# EVOLVE-BLOCK-END
