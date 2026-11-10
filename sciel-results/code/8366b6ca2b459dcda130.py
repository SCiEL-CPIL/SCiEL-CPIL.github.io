# EVOLVE-BLOCK-START
import time
import numpy as np


def _area_and_subgrad(x, a, y, w, n):
    """
    Vectorized over heights.
    Returns (area, subgradient w.r.t. x).
    L[m, i] = x_i + a_i*y_m ; width w_m = (1-y_m)/n.
    At each height, union = w + sum_j min(g_j, w), g_j = L_{j+1}-L_j over sorted L.
    Subgradient of min(g_j, w) wrt x_i = sign(L_i - L_{i+1}) if g_j < w else 0.
    """
    L = x[None, :] + a[None, :] * y[:, None]          # (M, n)
    order = np.argsort(L, axis=1)                      # (M, n)
    Ls = np.take_along_axis(L, order, axis=1)          # (M, n)
    g = np.diff(Ls, axis=1)                            # (M, n-1)
    union = w + np.minimum(g, w[:, None]).sum(axis=1)  # (M,)
    area = union.mean()
    active = g < w[:, None]                            # (M, n-1)
    sg = np.zeros(n)
    if active.any():
        p_idx = np.arange(n - 1)
        lo_idx = np.take_along_axis(order, p_idx[None, :], axis=1)        # (M, n-1)
        hi_idx = np.take_along_axis(order, (p_idx + 1)[None, :], axis=1)  # (M, n-1)
        np.add.at(sg, lo_idx[active], -1.0)
        np.add.at(sg, hi_idx[active], 1.0)
        sg /= y.size
    return area, sg


def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(20000) + 0.5) / 20000.0
    w = (1.0 - y) / n

    # Keich construction (score 1.0 baseline)
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)

    best_x = x.copy()
    best_val, _ = _area_and_subgrad(x, a, y, w, n)

    deadline = time.time() + 540.0
    LO, HI = -2.0, 2.0
    step = 0.02
    min_step = 1e-6
    shrink = 0.999
    no_improve = 0
    max_no_improve = 8000

    while time.time() < deadline:
        val, sg = _area_and_subgrad(x, a, y, w, n)
        gnorm = np.linalg.norm(sg)
        if gnorm < 1e-12:
            break
        x_new = x - step * sg / gnorm
        x_new = np.clip(x_new, LO, HI)
        val_new, _ = _area_and_subgrad(x_new, a, y, w, n)
        if val_new < best_val - 1e-12:
            best_val = val_new
            best_x = x_new.copy()
            x = x_new
            no_improve = 0
        else:
            x = x_new
            no_improve += 1
        step *= shrink
        if step < min_step:
            step = min_step
            if no_improve > max_no_improve:
                break

    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
