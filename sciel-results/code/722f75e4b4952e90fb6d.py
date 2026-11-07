# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 150.0
_MARGIN = 10.0


def _safe(v, lo=1e-9, hi=1.0 / 1e-9):
    v = np.asarray(v, dtype=float)
    return np.clip(np.nan_to_num(v, nan=0.0, posinf=hi, neginf=-hi), lo, hi)


def _basis(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x2 = x * x
    y2 = y * y
    x3 = x2 * x
    y3 = y2 * y
    xy = x * y
    x2y = x2 * y
    y2x = y2 * x
    x3y = x3 * y
    y3x = y3 * x
    x4 = x2 * x2
    y4 = y2 * y2
    x2y2 = x2 * y2
    sx = np.sin(x)
    sy = np.sin(y)
    cx = np.cos(x)
    cy = np.cos(y)
    sxy = np.sin(x + y)
    cxy = np.cos(x + y)
    sxd = np.sin(x - y)
    cxd = np.cos(x - y)
    ex = np.exp(_safe(x))
    ey = np.exp(_safe(y))
    # 10 terms
    return np.column_stack((
        np.ones_like(x),
        x,
        y,
        x2,
        y2,
        xy,
        x2y,
        y2x,
        x3y,
        y3x,
        x4,
        y4,
        x2y2,
        sx,
        sy,
        cx,
        cy,
        sxy,
        cxy,
        sxd,
        cxd,
        ex,
        ey,
    ))


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    if time.time() > _DEADLINE - _MARGIN:
        # Fallback: linear fit in closed form using first 3 params
        p = np.asarray(params, dtype=float)
        return p[0] * x + p[1] * y + p[2]

    B = _basis(x, y)
    p = np.asarray(params, dtype=float)
    pred = B @ p
    return pred
# EVOLVE-BLOCK-END
