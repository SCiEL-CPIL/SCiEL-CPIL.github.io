# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 630.0
_START = None


def _get_start():
    global _START
    if _START is None:
        _START = time.time()
    return _START


def _time_left():
    return _DEADLINE - (time.time() - _get_start())


def _keich_baseline(n=64):
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)


def _compute_area(x, n=64, n_heights=20000):
    """Compute union area via quadrature over n_heights vertical slices."""
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    # L_i = x_i + a_i * y  for each triangle at height y
    L = x[:, None] + a[:, None] * y[None, :]  # shape (n, n_heights)
    # w = (1 - y) / n, same for all triangles at a given height
    w = (1.0 - y) / n  # shape (n_heights,)
    # Sort L along axis 0
    L_sorted = np.sort(L, axis=0)
    # Union length = w + sum over consecutive pairs of min(L_{j+1} - L_j, w)
    gaps = np.diff(L_sorted, axis=0)  # shape (n-1, n_heights)
    union_len = w + np.sum(np.minimum(gaps, w[None, :]), axis=0)
    area = np.mean(union_len)
    return area


def _compute_area_fast(x, n=64, n_heights=20000):
    """Faster area computation using pre-allocated arrays."""
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1.0 - y) / n
    L = x[:, None] + a[:, None] * y[None, :]
    L_sorted = np.sort(L, axis=0)
    gaps = np.diff(L_sorted, axis=0)
    union_len = w + np.sum(np.minimum(gaps, w[None, :]), axis=0)
    return np.mean(union_len)


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem. For a fixed n, choose
    base positions x = (x_0, ..., x_{n-1}). For position index i (0-based) let
    a_i = (i + 1) / n and define the triangle R_i with vertices
        (x_i, 0), (x_i + 1/n, 0), (x_i + a_i, 1).
    Every triangle has base width 1/n on the line y = 0 and its apex on the line
    y = 1. The objective is to choose x so that the AREA of the union of the n
    triangles R_i is as small as possible. Here n = 64.

    Strategy: Start from the Keich baseline, then use scipy.optimize.minimize
    with SLSQP to refine positions. The union area is piecewise-linear and convex
    in the sorted interval starts, so global convex optimization can help escape
    local minima of coordinate descent.

    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    _get_start()
    n = 64
    x0 = _keich_baseline(n)
    best_x = x0.copy()
    best_area = _compute_area(x0, n)

    # Phase 1: SLSQP optimization from Keich baseline
    if _time_left() > 60:
        try:
            result = minimize(
                fun=_compute_area_fast,
                x0=x0,
                args=(n,),
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            )
            if result.success or (result.fun < best_area):
                best_x = result.x.copy()
                best_area = result.fun
        except Exception:
            pass

    # Phase 2: Polish with finer quadrature and tighter tolerance
    if _time_left() > 30:
        try:
            def _area_fine(x, n=64, n_heights=50000):
                a = (np.arange(n) + 1) / n
                y = (np.arange(n_heights) + 0.5) / n_heights
                w = (1.0 - y) / n
                L = x[:, None] + a[:, None] * y[None, :]
                L_sorted = np.sort(L, axis=0)
                gaps = np.diff(L_sorted, axis=0)
                union_len = w + np.sum(np.minimum(gaps, w[None, :]), axis=0)
                return np.mean(union_len)

            result2 = minimize(
                fun=_area_fine,
                x0=best_x,
                args=(n,),
                method='SLSQP',
                options={'maxiter': 100, 'ftol': 1e-14, 'disp': False}
            )
            if result2.success or (result2.fun < best_area):
                best_x = result2.x.copy()
                best_area = result2.fun
        except Exception:
            pass

    # Phase 3: Final coordinate-descent polish with very small steps
    if _time_left() > 15:
        try:
            x_curr = best_x.copy()
            step = 1e-5
            for _ in range(50):
                if _time_left() < 10:
                    break
                improved = False
                for i in range(n):
                    for delta in [-step, step]:
                        x_try = x_curr.copy()
                        x_try[i] += delta
                        area_try = _compute_area(x_try, n, n_heights=20000)
                        if area_try < best_area - 1e-15:
                            x_curr = x_try
                            best_area = area_try
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-9:
                        break
            if best_area < _compute_area(best_x, n):
                best_x = x_curr
        except Exception:
            pass

    return best_x


# EVOLVE-BLOCK-END
