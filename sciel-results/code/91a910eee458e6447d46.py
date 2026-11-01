"""Packing of 50 equal circles in a unit square via soft-min optimization + SA + binary search."""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import cKDTree

N = 50
HALF = 0.5
DEADLINE = 140.0  # Safety margin before 150s kill


def _compute_r(pts):
    """Compute the feasible radius for a configuration (double precision)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # boundary: min distance from each point to the square edge
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r = float(np.min(d))
    # pairwise using cKDTree for efficiency
    tree = cKDTree(pts)
    dists, idx = tree.query(pts, k=2)
    min_pair = float(np.min(dists[:, 1])) / 2.0
    r = min(r, min_pair)
    return r


def _soft_min_obj(x, tau=0.008):
    """Objective: minimize -soft_min(minimum constraint distance)."""
    centers = x.reshape(-1, 2)
    
    # Wall distances
    wall_x = HALF - np.abs(centers[:, 0])
    wall_y = HALF - np.abs(centers[:, 1])
    walls = np.minimum(wall_x, wall_y)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Upper triangle
    iu, ju = np.triu_indices(N, k=1)
    pair_upper = pair_dists[iu, ju]
    
    all_dists = np.concatenate([walls, pair_upper])
    
    # Soft-min via log-sum-exp
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    log_sum = np.log(np.sum(np.exp(-shifted / tau)))
    soft_min = min_d - tau * log_sum
    
    return -soft_min


def _soft_min_grad(x, tau=0.008):
    """Analytical gradient of the soft-min objective."""
    centers = x.reshape(-1, 2)
    
    wall_x = HALF - np.abs(centers[:, 0])
    wall_y = HALF - np.abs(centers[:, 1])
    walls = np.minimum(wall_x, wall_y)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    iu, ju = np.triu_indices(N, k=1)
    pair_upper = pair_dists[iu, ju]
    
    all_dists = np.concatenate([walls, pair_upper])
    
    # Soft-min weights
    min_d = np.min(all_dists)
    shifted = all_dists - min_d
    exp_vals = np.exp(-shifted / tau)
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(x)
    
    # Wall gradient contributions
    for i in range(N):
        w = weights[i]
        if walls[i] == wall_x[i] and wall_x[i] < wall_y[i]:
            grad[2*i] += w * np.sign(centers[i, 0])
        elif walls[i] == wall_y[i] and wall_y[i] <= wall_x[i]:
            grad[2*i+1] += w * np.sign(centers[i, 1])
        else:
            # Both equal, split
            grad[2*i] += 0.5 * w * np.sign(centers[i, 0])
            grad[2*i+1] += 0.5 * w * np.sign(centers[i, 1])
    
    # Pairwise gradient contributions
    for k in range(len(iu)):
        w = weights[N + k]
        i, j = iu[k], ju[k]
        d = dists[i, j]
        if d > 1e-12:
            factor = 0.5 * w / d
            diff_ij = centers[i] - centers[j]
            grad[2*i:2*i+2] -= factor * diff_ij
            grad[2*j:2*j+2] += factor * diff_ij
    
    return grad


def _violation(pts, r_target):
    """Total violation: sum of pairwise overlaps + wall penetrations."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_viol = np.maximum(0.0, 2.0 * r_target - dist)
    iu = np.triu_indices(N, k=1)
    pair_viol = pair_viol[iu]
    
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    wall_viol = np.maximum(0.0, r_target - d)
    
    return float(np.sum(pair_viol) + np.sum(wall_viol))


def _simulated_annealing(pts0, r_target, deadline, rng, n_steps=5000):
    """Simulated annealing to minimize violation for target radius r_target."""
    pts = pts0.copy()
    current_viol = _violation(pts, r_target)
    best_pts = pts.copy()
    best_viol = current_viol
    
    t_start = 0.05
    t_end = 0.0001
    cooling = (t_end / t_start) ** (1.0 / n_steps)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        temp = t_start * (cooling ** step)
        
        n_move = rng.integers(1, 4)
        indices = rng.choice(N, size=n_move, replace=False)
        deltas = rng.normal(0, temp, size=(n_move, 2))
        
        new_pts = pts.copy()
        new_pts[indices] += deltas
        new_pts = np.clip(new_pts, -HALF, HALF)
        
        new_viol = _violation(new_pts, r_target)
        
        delta = new_viol - current_viol
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            pts = new_pts
            current_viol = new_viol
            if current_viol < best_viol:
                best_viol = current_viol
                best_pts = pts.copy()
    
    return best_pts, best_viol


def _lbfgs_softmin(x0, tau=0.008, deadline=0.0, maxiter=300):
    """L-BFGS-B on the soft-min objective."""
    start_time = time.time()
    
    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        return _soft_min_obj(x, tau)
    
    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        return _soft_min_grad(x, tau)
    
    try:
        res = opt.minimize(obj, x0, jac=grad, method='L-BFGS-B',
                           bounds=[(-HALF, HALF)] * (2 * N),
                           options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10,
                                    'maxfun': 1500, 'disp': False})
        pts = res.x.reshape(N, 2)
        pts = np.clip(pts, -HALF, HALF)
        return pts
    except Exception:
        return x0.reshape(N, 2)


def _lbfgs_violation(pts0, r_target, deadline):
    """L-BFGS-B to minimize violation for target radius."""
    start_time = time.time()
    
    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return _violation(p, r_target)
    
    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        # Numerical gradient
        eps = 1e-7
        g = np.zeros_like(x)
        base = obj(x)
        for i in range(x.size):
            xp = x.copy()
            xp[i] += eps
            g[i] = (obj(xp) - base) / eps
        return g
    
    x0 = pts0.ravel()
    try:
        res = opt.minimize(obj, x0, method='L-BFGS-B',
                           jac=grad,
                           bounds=[(-HALF, HALF)] * (2 * N),
                           options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-8})
        pts = res.x.reshape(N, 2)
        pts = np.clip(pts, -HALF, HALF)
        return pts, float(res.fun)
    except Exception:
        return pts0, _violation(pts0, r_target)


def _hex_init(rng, jitter=0.0):
    """Hexagonal close-packed lattice initialization."""
    pts = []
    a = 0.12
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        while len(pts) < N and x < HALF - a + 1e-9:
            pts.append([x, -HALF + a + row * dy])
            x += dx
        row += 1
    pts = np.array(pts[:N])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF, HALF)


def _grid_init(rng, jitter=0.0):
    """Grid initialization."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    grid = np.vstack([grid, extra])[:N]
    if jitter > 0:
        grid = grid + rng.normal(0, jitter, size=grid.shape)
    return np.clip(grid, -HALF, HALF)


def _random_init(rng):
    """Random initialization."""
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))


def _binary_search_radius(pts_start, deadline, rng):
    """Binary search on radius with violation minimization + soft-min polish."""
    best_pts = pts_start.copy()
    best_r = _compute_r(pts_start)
    
    r_lo = best_r
    r_hi = 0.07138
    
    for _ in range(12):
        if time.time() > deadline - 5.0:
            break
        
        r_mid = (r_lo + r_hi) / 2.0
        
        # SA phase
        pts_sa, viol_sa = _simulated_annealing(best_pts, r_mid, deadline - 5.0, rng, n_steps=2500)
        
        # L-BFGS violation polish
        pts_lbfgs, viol_lbfgs = _lbfgs_violation(pts_sa, r_mid, deadline - 5.0)
        
        # Additional soft-min polish
        x = pts_lbfgs.ravel()
        pts_polished = _lbfgs_softmin(x, tau=0.005, deadline=deadline - 3.0, maxiter=200)
        
        r_achieved = _compute_r(pts_polished)
        
        if viol_lbfgs < 1e-6 and r_achieved >= r_mid - 1e-8:
            r_lo = r_mid
            best_pts = pts_polished.copy()
            best_r = r_achieved
        else:
            r_hi = r_mid
    
    return best_pts, best_r


def construct():
    start = time.time()
    deadline = start + DEADLINE
    rng = np.random.default_rng(12345)
    
    best_pts = None
    best_r = 0.0
    
    def try_improve(pts0, r_target, n_sa_steps):
        nonlocal best_pts, best_r
        if time.time() > deadline - 10.0:
            return
        
        # SA phase
        pts_sa, viol_sa = _simulated_annealing(pts0, r_target, deadline - 10.0, rng, n_steps=n_sa_steps)
        
        # L-BFGS violation polish
        try:
            pts_lbfgs, viol_lbfgs = _lbfgs_violation(pts_sa, r_target, deadline - 10.0)
        except Exception:
            pts_lbfgs = pts_sa
            viol_lbfgs = viol_sa
        
        # Soft-min polish for additional refinement
        x = pts_lbfgs.ravel()
        pts_polished = _lbfgs_softmin(x, tau=0.005, deadline=deadline - 5.0, maxiter=200)
        
        r = _compute_r(pts_polished)
        if r > best_r:
            best_r = r
            best_pts = pts_polished.copy()
    
    # Phase 1: Multi-start with moderate target
    inits = [
        _hex_init(rng, jitter=0.01),
        _hex_init(rng, jitter=0.03),
        _grid_init(rng, jitter=0.02),
        _grid_init(rng, jitter=0.05),
        _random_init(rng),
        _random_init(rng),
    ]
    
    for init in inits:
        if time.time() > deadline - 30.0:
            break
        try_improve(init, 0.07, 2000)
    
    if best_pts is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Phase 2: Binary search on radius
    if time.time() < deadline - 20.0:
        best_pts, best_r = _binary_search_radius(best_pts, deadline, rng)
    
    # Phase 3: Final polish with soft-min at slightly higher target
    if time.time() < deadline - 5.0:
        r_target = best_r * 1.002
        try_improve(best_pts, r_target, 3000)
    
    return best_pts
