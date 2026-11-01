"""
50 equal circles in a unit square – model-and-optimize approach.
Strategy:
  1. Start from a near-optimal hexagonal grid.
  2. Coarse-to-fine L-BFGS-B on a smooth soft-min surrogate of the minimum
     clearance (pair-to-pair and circle-to-wall), with analytic gradient.
  3. Repeat from several diverse initial placements to escape local minima.
  4. Final high-precision polish with very tight tolerances.
"""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50
DEADLINE = 145.0  # safety margin before 150 s kill

# ---------------------------------------------------------------------------
# Radius computation (exact, used for scoring)
# ---------------------------------------------------------------------------
def _compute_radius(centers):
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    wall_x = 0.5 - np.abs(centers[:, 0])
    wall_y = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wall_x, wall_y))
    if min_wall < 0:
        return 0.0
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1]) / 2.0
    return min(min_pair, min_wall)

# ---------------------------------------------------------------------------
# Smooth objective:  -soft_min(all clearances)
#   all clearances = {wall_x_i, wall_y_i} ∪ {‖c_i-c_j‖/2 for i<j}
#   soft_min(d; τ) = min(d) - τ·log(Σ exp(-(d_k - min(d))/τ))
# ---------------------------------------------------------------------------
def _objective_and_grad(x, tau):
    """Return (f, g) where f = -soft_min, g = gradient."""
    c = x.reshape(-1, 2)

    # Wall clearances
    wx = 0.5 - np.abs(c[:, 0])
    wy = 0.5 - np.abs(c[:, 1])
    walls = np.minimum(wx, wy)

    # Pairwise clearances (upper triangle)
    iu, ju = np.triu_indices(N, k=1)
    diff = c[iu] - c[ju]
    pd = np.sqrt(np.sum(diff**2, axis=1)) / 2.0

    all_d = np.concatenate([walls, pd])
    min_d = np.min(all_d)
    shifted = all_d - min_d
    exp_vals = np.exp(-shifted / tau)
    sum_exp = np.sum(exp_vals)
    w = exp_vals / sum_exp
    soft_min = min_d - tau * np.log(sum_exp)

    grad = np.zeros_like(x)
    g = grad.reshape(N, 2)

    # Wall gradients
    mask_x = wx <= wy
    mask_y = wy <= wx
    
    # d(-soft_min)/dx = w * sign(x)
    # Only add if active
    if np.any(mask_x):
        g[mask_x, 0] += w[mask_x] * np.sign(c[mask_x, 0])
    if np.any(mask_y):
        g[mask_y, 1] += w[mask_y] * np.sign(c[mask_y, 1])

    # Pair gradients
    # d(-soft_min)/dc_i = - w_k * 0.5 * (c_i - c_j) / ||c_i - c_j||
    # ||c_i - c_j|| = 2 * pd
    # term = - 0.25 * w_k / pd * (c_i - c_j)
    
    valid = pd > 1e-12
    if np.any(valid):
        w_pair = w[N:]
        factors = np.zeros_like(pd)
        factors[valid] = 0.25 * w_pair[valid] / pd[valid]
        
        # diff is (M, 2)
        # We want to subtract factor * diff from c_i and add to c_j
        # np.add.at(g, iu, -factors[:, None] * diff)
        # np.add.at(g, ju, factors[:, None] * diff)
        
        # Since iu and ju are arrays of indices, and factors/diff are aligned,
        # we can use np.add.at
        np.add.at(g, iu, -factors[:, None] * diff)
        np.add.at(g, ju, factors[:, None] * diff)

    return -soft_min, grad

# ---------------------------------------------------------------------------
# Initial placements
# ---------------------------------------------------------------------------
def _hex_grid(nrows, ncols, offset_rows=True):
    """Hexagonal grid of nrows*ncols points in [-0.5,0.5]^2."""
    pts = []
    dx = 1.0 / (ncols + 0.5)
    dy = dx * np.sqrt(3) / 2.0
    for r in range(nrows):
        y = -0.5 + dy * (r + 0.5)
        if y > 0.5:
            break
        off = dx / 2.0 if (offset_rows and r % 2 == 1) else 0.0
        for c in range(ncols):
            x = -0.5 + dx * (c + 0.5) + off
            if -0.5 <= x <= 0.5:
                pts.append([x, y])
    return np.array(pts[:N])

def _square_grid():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = []
    for y in ticks:
        for x in ticks:
            pts.append([x, y])
            if len(pts) >= N:
                return np.array(pts)
    return np.array(pts[:N])

def _jitter(pts, rng, sigma=0.005):
    p = pts + rng.normal(0, sigma, pts.shape)
    return np.clip(p, -0.49, 0.49)

# ---------------------------------------------------------------------------
# L-BFGS-B optimizer with time budget
# ---------------------------------------------------------------------------
def _lbfgs(x0, tau, budget, maxiter=2000, ftol=1e-14, gtol=1e-12):
    t0 = time.time()
    def fg(v):
        if time.time() - t0 > budget:
            return 0.0, np.zeros_like(v)
        return _objective_and_grad(v, tau)
    try:
        res = opt.minimize(fg, x0, jac=True, method='L-BFGS-B',
                           bounds=[(-0.5, 0.5)] * (2*N),
                           options={'maxiter': maxiter, 'ftol': ftol,
                                    'gtol': gtol, 'maxfun': maxiter * 2,
                                    'disp': False})
        return np.clip(res.x.reshape(-1, 2), -0.5, 0.5)
    except Exception:
        return x0.reshape(-1, 2)

# ---------------------------------------------------------------------------
# Multi-restart coarse-to-fine
# ---------------------------------------------------------------------------
def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE

    rng = np.random.default_rng(42)

    # Generate diverse initial placements
    inits = []
    # Hex grids with different row/col counts
    for nrows, ncols in [(7, 8), (8, 7), (7, 7), (6, 9), (9, 6)]:
        g = _hex_grid(nrows, ncols)
        if len(g) >= N:
            inits.append(_jitter(g, rng, 0.004))
    # Square grid
    inits.append(_jitter(_square_grid(), rng, 0.004))
    # Random
    inits.append(_jitter(rng.uniform(-0.45, 0.45, (N, 2)), rng, 0.0))

    best_c = inits[0]
    best_r = _compute_radius(best_c)

    # Coarse-to-fine tau schedule
    taus = [0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001, 0.0005]

    for init in inits:
        if time.time() > deadline - 10:
            break
        x = init.flatten().copy()
        for tau in taus:
            if time.time() > deadline - 8:
                break
            remaining = deadline - time.time() - 6
            if remaining < 2:
                break
            # Use more iterations if tau is small
            max_iter = 3000 if tau < 0.01 else 1500
            x = _lbfgs(x, tau, budget=remaining, maxiter=max_iter)
            r = _compute_radius(x.reshape(-1, 2))
            if r > best_r:
                best_r = r
                best_c = x.reshape(-1, 2).copy()

    # Final high-precision polish
    if time.time() < deadline - 3:
        remaining = deadline - time.time() - 2
        x = best_c.flatten()
        for tau in [0.0005, 0.0002, 0.0001, 0.00005]:
            if time.time() > deadline - 2:
                break
            x = _lbfgs(x, tau, budget=remaining, maxiter=5000,
                       ftol=1e-16, gtol=1e-14)
            r = _compute_radius(x.reshape(-1, 2))
            if r > best_r:
                best_r = r
                best_c = x.reshape(-1, 2).copy()

    if not np.all(np.isfinite(best_c)) or best_c.shape != (N, 2):
        return _hex_grid(7, 8)
    return best_c
