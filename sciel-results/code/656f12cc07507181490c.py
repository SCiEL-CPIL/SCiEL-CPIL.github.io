# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _union_length_1d_fast(x, N=8, n_samples=50000):
    """
    Fast Monte Carlo estimate of the union length of N intervals of width 1/N,
    where interval i is [x_i + c*i/N, x_i + c*i/N + 1/N] for c in [0,1].
    Uses a fixed deterministic sample of c values for speed and reproducibility.
    """
    c = np.linspace(0.0, 1.0, n_samples, endpoint=False)
    shifts = c[:, None] * np.arange(N)[None, :] / N
    starts = x[None, :] + shifts.T  # (N, n_samples)
    ends = starts + 1.0 / N
    order = np.argsort(starts, axis=0)
    sorted_starts = np.take_along_axis(starts, order, axis=0)
    sorted_ends = np.take_along_axis(ends, order, axis=0)
    current_right = sorted_ends[0].copy()
    union_len = np.zeros(n_samples)
    for i in range(N):
        s_i = sorted_starts[i]
        e_i = sorted_ends[i]
        mask = s_i <= current_right
        union_len += np.where(mask, np.minimum(e_i, current_right) - s_i, e_i - s_i)
        current_right = np.where(mask, np.maximum(current_right, e_i), e_i)
    return union_len.mean()


def _keich_initial(N=8):
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 580.0

    keich = _keich_initial(N)

    # Generate restarts
    rng_init = np.random.default_rng(42)
    restarts = [keich.copy()]
    for _ in range(5):
        pert = keich + rng_init.normal(0, 0.1, N)
        restarts.append(pert)

    # Optimize x-positions
    best_x = keich.copy()
    best_x_obj = _union_length_1d_fast(best_x)

    for x0 in restarts:
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(_union_length_1d_fast, x0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
            if np.isfinite(res.fun) and res.fun < best_x_obj:
                best_x_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Optimize y-positions
    best_y = keich.copy()
    best_y_obj = _union_length_1d_fast(best_y)

    for y0 in restarts:
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(_union_length_1d_fast, y0, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
            if np.isfinite(res.fun) and res.fun < best_y_obj:
                best_y_obj = res.fun
                best_y = res.x.copy()
        except Exception:
            pass

    # Build positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = best_x[i_flat]
    base_y = best_y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)

    # Validate
    x = positions[:, 0]
    y = positions[:, 1]
    i_idx = i_flat
    j_idx = j_flat
    X0 = np.min(x)
    X1 = np.max(x + (i_idx + 1) / N)
    Y0 = np.min(y)
    Y1 = np.max(y + (j_idx + 1) / N)

    if not (np.all(np.isfinite(positions)) and (X1 - X0) > 1e-9 and (Y1 - Y0) > 1e-9):
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)

    return positions


# EVOLVE-BLOCK-END
