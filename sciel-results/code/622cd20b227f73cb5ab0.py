"""
50 equal circles in a unit square – model-and-optimize approach.
Strategy:
  1. Start from a near-optimal hexagonal grid.
  2. Coarse-to-fine L-BFGS-B on a smooth soft-min surrogate of the minimum
     clearance (pair-to-pair and circle-to-wall), with analytic gradient.
  3. Simulated annealing for global exploration between local refinements.
  4. Repeat from several diverse initial placements to escape local minima.
  5. Final high-precision polish with very tight tolerances.
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
    w = exp_vals / np.sum(exp_vals)
    soft_min = min_d - tau * np.log(np.sum(exp_vals))

    grad = np.zeros_like(x)

    # Wall gradients
    for i in range(N):
        wi = w[i]
        if wx[i] <= wy[i]:
            grad[2*i]   += wi * np.sign(c[i, 0])
        if wy[i] <= wx[i]:
            grad[2*i+1] += wi * np.sign(c[i, 1])

    # Pair gradients
    for k in range(len(iu)):
        wk = w[N + k]
        i, j = iu[k], ju[k]
        dvec = c[i] - c[j]
        d = np.sqrt(np.dot(dvec, dvec))
        if d > 1e-12:
            f = 0.5 * wk / d
            grad[2*i]   -= f * dvec[0]
            grad[2*i+1] -= f * dvec[1]
            grad[2*j]   += f * dvec[0]
            grad[2*j+1] += f * dvec[1]

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
# Simulated annealing for global exploration
# ---------------------------------------------------------------------------
def _simulated_annealing(x0, budget, rng):
    """Run simulated annealing to explore the search space."""
    t0 = time.time()
    x = x0.copy()
    current_r = _compute_radius(x.reshape(-1, 2))
    best_x = x.copy()
    best_r = current_r
    
    # Annealing parameters
    T_start = 0.05
    T_end = 0.001
    n_iter = 5000
    
    for it in range(n_iter):
        if time.time() - t0 > budget:
            break
        
        T = T_start * (T_end / T_start) ** (it / n_iter)
        
        # Propose a random move
        idx = rng.integers(0, N)
        step_size = 0.02 * (1.0 - 0.5 * it / n_iter)
        delta = rng.normal(0, step_size, 2)
        x_new = x.copy()
        x_new[idx] += delta
        x_new = np.clip(x_new, -0.5, 0.5)
        
        new_r = _compute_radius(x_new.reshape(-1, 2))
        
        # Accept with probability exp(-(new_r - current_r)/T) if worse
        if new_r >= current_r or rng.random() < np.exp(-(new_r - current_r) / T):
            x = x_new
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_x = x.copy()
    
    return best_x.reshape(-1, 2)

# ---------------------------------------------------------------------------
# Multi-restart coarse-to-fine with simulated annealing
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
        if time.time() > deadline - 15:
            break
        x = init.flatten().copy()
        
        # Phase 1: Coarse L-BFGS
        for tau in [0.03, 0.02, 0.015, 0.01]:
            if time.time() > deadline - 12:
                break
            remaining = deadline - time.time() - 10
            if remaining < 2:
                break
            x = _lbfgs(x, tau, budget=remaining, maxiter=1000)
            r = _compute_radius(x.reshape(-1, 2))
            if r > best_r:
                best_r = r
                best_c = x.reshape(-1, 2).copy()
        
        # Phase 2: Simulated annealing for exploration
        if time.time() < deadline - 8:
            remaining_sa = deadline - time.time() - 7
            if remaining_sa > 5:
                x_sa = _simulated_annealing(x.reshape(-1, 2), budget=remaining_sa, rng=rng)
                r_sa = _compute_radius(x_sa)
                if r_sa > best_r:
                    best_r = r_sa
                    best_c = x_sa.copy()
                x = x_sa.flatten()
        
        # Phase 3: Fine L-BFGS
        for tau in [0.007, 0.005, 0.003, 0.002, 0.001, 0.0005]:
            if time.time() > deadline - 6:
                break
            remaining = deadline - time.time() - 5
            if remaining < 1:
                break
            x = _lbfgs(x, tau, budget=remaining, maxiter=1500)
            r = _compute_radius(x.reshape(-1, 2))
            if r > best_r:
                best_r = r
                best_c = x.reshape(-1, 2).copy()

    # Final high-precision polish
    if time.time() < deadline - 3:
        remaining = deadline - time.time() - 2
        x = best_c.flatten()
        for tau in [0.0005, 0.0002, 0.0001]:
            if time.time() > deadline - 2:
                break
            x = _lbfgs(x, tau, budget=remaining, maxiter=3000,
                       ftol=1e-16, gtol=1e-14)
            r = _compute_radius(x.reshape(-1, 2))
            if r > best_r:
                best_r = r
                best_c = x.reshape(-1, 2).copy()

    if not np.all(np.isfinite(best_c)) or best_c.shape != (N, 2):
        return _hex_grid(7, 8)
    return best_c
