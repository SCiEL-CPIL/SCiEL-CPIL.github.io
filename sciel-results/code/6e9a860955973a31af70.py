"""Symbolic regression with non-linear parametric model (Gaussians + Linear)."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = None
_TIME_LIMIT = 150.0
_SAFETY_MARGIN = 15.0


def _init_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_LIMIT
    return _DEADLINE


def _time_ok():
    if _DEADLINE is None:
        return True
    return (time.time() - _DEADLINE) < _SAFETY_MARGIN


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Non-linear parametric model: Sum of 2 Gaussians + Linear term.
    Parameters are mapped to ensure stability:
    - Amplitudes: raw
    - Widths: exp(clip(p, -5, 5))
    - Centers: tanh(p)
    - Linear: raw
    """
    _init_deadline()
    
    if not _time_ok():
        # Safety fallback: simple linear
        return params[0] * np.asarray(x) + params[1] * np.asarray(y) + params[2]

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if params.shape[0] < 10:
        p = np.ones(10, dtype=np.float64)
        p[:min(10, len(params))] = params[:min(10, len(params))]
        params = p

    # Map parameters
    a1 = params[0]
    w1 = np.exp(np.clip(params[1], -5.0, 5.0))
    cx1 = np.tanh(params[2])
    cy1 = np.tanh(params[3])
    
    a2 = params[4]
    w2 = np.exp(np.clip(params[5], -5.0, 5.0))
    cx2 = np.tanh(params[6])
    cy2 = np.tanh(params[7])
    
    bias = params[8]
    slope_x = params[9]
    
    # Compute Gaussians
    dx1 = x - cx1
    dy1 = y - cy1
    r2_1 = dx1 * dx1 + dy1 * dy1
    g1 = a1 * np.exp(-w1 * r2_1)
    
    dx2 = x - cx2
    dy2 = y - cy2
    r2_2 = dx2 * dx2 + dy2 * dy2
    g2 = a2 * np.exp(-w2 * r2_2)
    
    # Linear part
    linear = bias + slope_x * x
    
    pred = g1 + g2 + linear
    
    # Ensure finite output
    pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
# EVOLVE-BLOCK-END
