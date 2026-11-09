"""
Mixed polynomial-trigonometric-radial basis for symbolic regression.
Combines the structural idea of the current program (polynomial + trig)
with the refinement machinery of inspiration 2 (radial features, deadline
guard, numerical safety). The 10 basis functions span:
  - Constant and linear terms
  - Quadratic polynomial terms (x^2, y^2, x*y)
  - Trigonometric terms (sin(x), sin(y))
  - Radial distance r = sqrt(x^2 + y^2)
  - Periodic interaction cos(x*y)
This diverse hypothesis space captures both polynomial and periodic
structure in the hidden function.
"""
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
    """Closed-form prediction using 10 learned parameters."""
    _init_deadline()
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    if params.shape[0] < 10:
        p = np.ones(10, dtype=np.float64)
        p[:min(10, len(params))] = params[:min(10, len(params))]
        params = p
    
    if not _time_ok():
        # Safety fallback: simple linear combination
        return params[0] + params[1] * x + params[2] * y
    
    n = len(x)
    xc = np.clip(x, -1.0, 1.0)
    yc = np.clip(y, -1.0, 1.0)
    
    # Compute the 10 basis features
    t0 = np.ones(n)
    t1 = xc
    t2 = yc
    t3 = xc * xc
    t4 = yc * yc
    t5 = xc * yc
    t6 = np.sin(xc)
    t7 = np.sin(yc)
    r = np.sqrt(xc * xc + yc * yc)
    t8 = r
    t9 = np.cos(xc * yc)
    
    # Stack into feature matrix (n, 10)
    F = np.column_stack([t0, t1, t2, t3, t4, t5, t6, t7, t8, t9])
    
    # Ensure finite values
    F = np.nan_to_num(F, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Linear combination
    pred = F @ params[:10]
    
    # Ensure finite output
    pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return pred
# EVOLVE-BLOCK-END
