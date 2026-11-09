"""Symbolic regression for nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator.
    
    Uses a rich basis of trigonometric and polynomial terms to capture
    nonlinear dynamics with time-dependent forcing.
    
    params: 10 fitted constants
        p0: constant offset
        p1: linear in t
        p2: sin(t)
        p3: cos(t)
        p4: sin(2t)
        p5: cos(2t)
        p6: x * sin(t)
        p7: x * cos(t)
        p8: v * sin(t)
        p9: v * cos(t)
    """
    # Ensure inputs are 1D arrays
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Precompute trigonometric basis
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    sin_2t = np.sin(2.0 * t)
    cos_2t = np.cos(2.0 * t)
    
    # Build the model as a linear combination of basis functions
    # This keeps the problem convex in params, making BFGS robust
    result = (
        params[0] * np.ones_like(t) +
        params[1] * t +
        params[2] * sin_t +
        params[3] * cos_t +
        params[4] * sin_2t +
        params[5] * cos_2t +
        params[6] * (x * sin_t) +
        params[7] * (x * cos_t) +
        params[8] * (v * sin_t) +
        params[9] * (v * cos_t)
    )
    
    return result
# EVOLVE-BLOCK-END
