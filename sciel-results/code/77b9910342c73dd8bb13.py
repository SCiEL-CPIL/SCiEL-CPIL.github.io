"""Circle packing in perimeter-4 rectangle: Multi-start SLSQP with penalty polish.

Strategy:
  1. Iterate over a set of aspect ratios (w + h = 2).
  2. For each aspect ratio, generate multiple seeds (grid, random).
  3. Run SLSQP on (radii, centers) with explicit wall + non-overlap constraints.
  4. Run a final penalty-based minimization to ensure feasibility and squeeze radii.
  5. Return best feasible packing.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
_START = time.time()
DEADLINE = _START + 132.0  # safety margin before 150 s kill


def _time_ok(margin=0.0):
    return time.time() < DEADLINE - margin


def _penalty_objective(x, w, h, mu=1000.0):
    """Return penalty + objective for x=[r, cx, cy]."""
    r = x[:N]
    cx = x[N:2 * N]
    cy = x[2 * N:3 * N]
    if w <= 0.1 or h <= 0.1:
        return 1e6

    # Wall violations
    d_min = np.minimum(np.minimum(cx, cy), np.minimum(w - cx, h - cy))
    viol_wall = np.maximum(0.0, r - d_min)
    pen = mu * np.sum(viol_wall ** 2)

    # Pairwise overlap violations
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2
            s = r[i] + r[j]
            if d2 < s * s:
                pen += mu * (s * s - d2)
    
    # Objective: minimize negative sum of radii
    return pen - np.sum(r)


def _check_validity(centers, w, h, radii, tol=1e-7):
    for i in range(N):
        xi, yi = centers[i]
        ri = radii[i]
        if ri < 1e-9:
            return False
        if xi < -tol or xi > w + tol or yi < -tol or yi > h + tol:
            return False
        if ri > min(xi, w - xi, yi, h - yi) + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
            if d < radii[i] + radii[j] - tol:
                return False
    return True


def _grid_seed(w, h):
    """Grid seed x=[r, cx, cy] for SLSQP. Returns best feasible grid."""
    best_seed = None
    best_score = -1e9
    for cols in range(1, N + 1):
        rows = (N + cols - 1) // cols
        if cols * rows < N:
            continue
        cw = w / cols
        ch = h / rows
        r = 0.5 * min(cw, ch) * 0.999
        x = np.zeros(3 * N)
        idx = 0
        for j in range(rows):
            for i in range(cols):
                if idx >= N:
                    break
                x[idx] = r
                x[N + idx] = (i + 0.5) * cw
                x[2 * N + idx] = (j + 0.5) * ch
                idx += 1
        if _check_validity(x[N:2 * N], w, h, x[:N], tol=1e-6):
            s = float(np.sum(x[:N]))
            if s > best_score:
                best_score = s
                best_seed = x.copy()
    if best_seed is None:
        # fallback: small circles in a row
        x = np.zeros(3 * N)
        x[:N] = 1e-3
        xs = np.linspace(1e-3, w - 1e-3, N)
        x[N:2 * N] = xs
        x[2 * N:3 * N] = h / 2.0
        best_seed = x
    return best_seed


def _random_seed(w, h):
    """Random seed x=[r, cx, cy]."""
    x = np.zeros(3 * N)
    # Small initial radii
    x[:N] = 0.01
    # Random centers
    x[N:2 * N] = np.random.uniform(0.05, w - 0.05, N)
    x[2 * N:3 * N] = np.random.uniform(0.05, h - 0.05, N)
    return x


def _slsqp_optimize(x0, w, h, time_budget=30.0):
    """SLSQP on (r, cx, cy) with wall + non-overlap constraints."""
    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - 1e-5})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[N + i] - x[i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: (w - x[N + i]) - x[i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * N + i] - x[i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: (h - x[2 * N + i]) - x[i]})
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (
                    (x[N + i] - x[N + j]) ** 2 +
                    (x[2 * N + i] - x[2 * N + j]) ** 2 -
                    (x[i] + x[j]) ** 2
                ),
            })

    bounds = []
    for i in range(N):
        bounds.append((1e-5, min(w, h) / 2.0))
    for i in range(N):
        bounds.append((0.0, w))
    for i in range(N):
        bounds.append((0.0, h))

    start = time.time()
    max_iter = 400 if time_budget > 25 else 200
    try:
        res = minimize(
            lambda x: -np.sum(x[:N]),
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': max_iter, 'ftol': 1e-12},
        )
    except Exception:
        return x0
    x = res.x
    x[:N] = np.maximum(x[:N], 1e-5)
    return x


def _penalty_polish(x, w, h, time_budget=10.0):
    """Penalty-based polish using minimize."""
    start = time.time()
    if time.time() - start > time_budget:
        return x
    
    try:
        res = minimize(
            lambda x: _penalty_objective(x, w, h, mu=1000.0),
            x,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8},
        )
        if res.fun < _penalty_objective(x, w, h, mu=1000.0):
            return res.x
    except Exception:
        pass
    return x


def construct():
    best_val = -1e9
    best_arr = None

    aspect_ratios = [1.0, 1.2, 0.85, 1.3, 0.75, 1.1, 0.9, 1.4, 0.7, 1.25, 0.8]

    for w0 in aspect_ratios:
        if not _time_ok(30.0):
            break
        if w0 <= 0.2 or w0 >= 1.8:
            continue
        h0 = 2.0 - w0

        # Try grid seed
        seed_x = _grid_seed(w0, h0)
        
        # Try a few random seeds
        seeds = [seed_x]
        for _ in range(3):
            if not _time_ok(10.0):
                break
            seeds.append(_random_seed(w0, h0))

        for seed in seeds:
            if not _time_ok(15.0):
                break
            
            slsqp_x = _slsqp_optimize(seed, w0, h0, time_budget=20.0)
            slsqp_x = _penalty_polish(slsqp_x, w0, h0, time_budget=5.0)

            if _check_validity(slsqp_x[N:2 * N], w0, h0, slsqp_x[:N]):
                val = float(np.sum(slsqp_x[:N]))
                if val > best_val:
                    best_val = val
                    best_arr = np.column_stack([slsqp_x[N:2 * N], slsqp_x[2 * N:3 * N], slsqp_x[:N]])

    # Fallback
    if best_arr is None:
        w = h = 1.0
        seed_x = _grid_seed(w, h)
        slsqp_x = _slsqp_optimize(seed_x, w, h, time_budget=20.0)
        if not _check_validity(slsqp_x[N:2 * N], w, h, slsqp_x[:N]):
            # trivial valid
            xs = np.linspace(1e-3, w - 1e-3, N)
            ys = np.full(N, h / 2.0)
            r = np.full(N, 1e-3)
            best_arr = np.column_stack([xs, ys, r])
        else:
            best_arr = np.column_stack([slsqp_x[N:2 * N], slsqp_x[2 * N:3 * N], slsqp_x[:N]])

    out = np.array(best_arr[:N])
    return out
