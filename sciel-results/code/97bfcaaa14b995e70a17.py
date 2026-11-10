# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, quad


def _union_length_1d(starts):
    """Compute the length of the union of intervals [starts[i], starts[i]+w] for w=1/8."""
    n = len(starts)
    if n == 0:
        return 0.0
    order = np.argsort(starts)
    s_sorted = starts[order]
    w = 1.0 / 8.0
    total = 0.0
    cur_start = s_sorted[0]
    cur_end = cur_start + w
    for k in range(1, n):
        s_k = s_sorted[k]
        e_k = s_k + w
        if s_k <= cur_end:
            if e_k > cur_end:
                cur_end = e_k
        else:
            total += cur_end - cur_start
            cur_start = s_k
            cur_end = e_k
    total += cur_end - cur_start
    return total


def _volume_from_bases(x_bases, y_bases, c_grid=None, w_grid=None):
    """Compute the union volume given 8 x-bases and 8 y-bases using quadrature."""
    N = 8
    if c_grid is None:
        c_grid, w_grid = np.polynomial.legendre.leggauss(20)
        c_grid = 0.5 * (c_grid + 1)  # Map [-1, 1] to [0, 1]
        w_grid = 0.5 * w_grid
        
    vol = 0.0
    for c, w in zip(c_grid, w_grid):
        x_starts = x_bases + c * np.arange(N) / N
        y_starts = y_bases + c * np.arange(N) / N
        lx = _union_length_1d(x_starts)
        ly = _union_length_1d(y_starts)
        vol += w * lx * ly
    return vol


def _build_positions(x_bases, y_bases):
    """Build the (64, 2) array from 8 x-bases and 8 y-bases."""
    N = 8
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    base_x = x_bases[i_flat]
    base_y = y_bases[j_flat]
    return np.stack([base_x, base_y], axis=1)


def _validate_and_volume(positions):
    """Check validity and compute the exact Monte Carlo volume as specified."""
    if not np.all(np.isfinite(positions)):
        return None
    x_k = positions[:, 0]
    y_k = positions[:, 1]
    N = 8
    idx = np.arange(64)
    i_idx = idx // N
    j_idx = idx % N
    X0 = np.min(x_k)
    X1 = np.max(x_k + (i_idx + 1) / N)
    Y0 = np.min(y_k)
    Y1 = np.max(y_k + (j_idx + 1) / N)
    area = (X1 - X0) * (Y1 - Y0)
    if area <= 1e-9:
        return None
    rng = np.random.default_rng(12345)
    a = rng.uniform(X0, X1, 500000)
    b = rng.uniform(Y0, Y1, 500000)
    c = rng.uniform(0, 1, 500000)
    inside = np.zeros(500000, dtype=bool)
    for k in range(64):
        ik = i_idx[k]
        jk = j_idx[k]
        xk = x_k[k]
        yk = y_k[k]
        x_lo = xk + c * ik / N
        x_hi = x_lo + 1.0 / N
        y_lo = yk + c * jk / N
        y_hi = y_lo + 1.0 / N
        in_tube = (a >= x_lo) & (a <= x_hi) & (b >= y_lo) & (b <= y_hi)
        inside |= in_tube
    frac = np.mean(inside)
    V = frac * area
    if V <= 0:
        return None
    return V


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 30.0

    # Keich binary construction as initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    best_x = keich.copy()
    best_y = keich.copy()
    best_volume = _volume_from_bases(best_x, best_y)

    # Quadrature grid for optimization
    c_grid, w_grid = np.polynomial.legendre.leggauss(32)
    c_grid = 0.5 * (c_grid + 1)
    w_grid = 0.5 * w_grid

    def obj_x(x):
        return _volume_from_bases(x, best_y, c_grid, w_grid)

    def obj_y(y):
        return _volume_from_bases(best_x, y, c_grid, w_grid)

    # Multi-start alternating minimization
    n_restarts = 10
    rng_opt = np.random.default_rng(42)
    
    for r in range(n_restarts):
        if time.time() > deadline - safety_margin:
            break

        if r == 0:
            x_curr = keich.copy()
            y_curr = keich.copy()
        else:
            # Perturb best
            x_curr = best_x + rng_opt.normal(0, 0.05, N)
            y_curr = best_y + rng_opt.normal(0, 0.05, N)

        # Alternate minimization
        for step in range(20):
            if time.time() > deadline - safety_margin:
                break
            
            # Optimize x
            try:
                res_x = minimize(obj_x, x_curr, method='L-BFGS-B', 
                                 options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                x_curr = res_x.x
            except Exception:
                pass
            
            # Optimize y
            try:
                res_y = minimize(obj_y, y_curr, method='L-BFGS-B',
                                 options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                y_curr = res_y.x
            except Exception:
                pass

        # Evaluate current solution
        vol_curr = _volume_from_bases(x_curr, y_curr, c_grid, w_grid)
        if vol_curr < best_volume:
            best_x = x_curr.copy()
            best_y = y_curr.copy()
            best_volume = vol_curr

    # Final validation with the exact Monte Carlo method
    best_positions = _build_positions(best_x, best_y)
    final_V = _validate_and_volume(best_positions)
    if final_V is None:
        # Fallback
        best_positions = _build_positions(keich, keich)

    return best_positions


# EVOLVE-BLOCK-END
