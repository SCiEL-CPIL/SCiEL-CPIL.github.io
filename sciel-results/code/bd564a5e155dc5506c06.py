"""Improved sphere packing via Lloyd equalization + soft-min polish + MC + local search."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    dist_to_wall = np.min(0.5 - np.abs(centers), axis=1)
    r_wall = np.min(dist_to_wall)
    
    n = len(centers)
    if n < 2:
        return max(0, r_wall)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    
    return min(r_wall, r_pair)

def _pair_dist_matrix(centers):
    """Compute pairwise distance matrix."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    return dists

def _lloyd_step(centers, n_iters=5):
    """Voronoi-Lloyd iteration: move each point toward centroid of nearest neighbors."""
    centers = centers.copy()
    for _ in range(n_iters):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        new_centers = centers.copy()
        for i in range(N):
            k = min(5, N-1)
            nearest_idx = np.argsort(dists[i])[:k]
            weights = 1.0 / dists[i, nearest_idx]
            weights /= np.sum(weights)
            target = np.average(centers[nearest_idx], axis=0, weights=weights)
            new_centers[i] = 0.7 * centers[i] + 0.3 * target
            for d in range(3):
                if new_centers[i, d] > 0.45:
                    new_centers[i, d] = 0.45
                elif new_centers[i, d] < -0.45:
                    new_centers[i, d] = -0.45
        centers = new_centers
    return centers

def _lloyd_optimize(centers, time_budget=12.0):
    """Run Lloyd iterations with periodic radius evaluation."""
    start = time.time()
    best = centers.copy()
    best_r = _compute_radius(centers)
    
    for it in range(100):
        if time.time() - start > time_budget:
            break
        centers = _lloyd_step(centers, n_iters=5)
        centers = np.clip(centers, -0.49, 0.49)
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best = centers.copy()
    return best

def _softmin_radius(centers, alpha):
    """Compute soft-min of all constraint values."""
    dist_to_wall = 0.5 - np.abs(centers)
    min_wall = np.min(dist_to_wall, axis=1)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    half_dists = dists / 2.0
    np.fill_diagonal(half_dists, 1e6)
    
    pair_vals = half_dists[~np.eye(N, dtype=bool)]
    all_vals = np.concatenate([min_wall, pair_vals])
    
    a = -alpha
    shifted = a * all_vals
    max_shift = np.max(shifted)
    log_sum_exp = max_shift + np.log(np.sum(np.exp(shifted - max_shift)))
    soft_min = -log_sum_exp / alpha
    return soft_min

def _smooth_obj(centers, alpha):
    return -_softmin_radius(centers, alpha)

def _smooth_polish(centers, alpha, time_budget=6.0):
    """Polish using L-BFGS-B with smooth objective."""
    start = time.time()
    try:
        res = minimize(
            _smooth_obj,
            centers,
            args=(alpha,),
            method='L-BFGS-B',
            bounds=[(-0.499, 0.499)] * (N * 3),
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            return res.x
    except Exception:
        pass
    return centers

def _repulsion_anneal(centers, time_budget=15.0):
    """Vectorized repulsion annealing."""
    start = time.time()
    best = centers.copy()
    best_r = _compute_radius(centers)
    
    step = 0.03
    for it in range(300):
        if time.time() - start > time_budget:
            break
        step *= 0.99
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        forces = np.zeros_like(centers)
        for i in range(N):
            for j in range(i+1, N):
                d = dists[i, j]
                if d < 0.2 and d > 1e-10:
                    f = step * (0.2 - d) / (d * d)
                    forces[i] += f * diff[i, j]
                    forces[j] -= f * diff[i, j]
        
        centers = centers + forces
        centers = np.clip(centers, -0.49, 0.49)
        
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best = centers.copy()
    
    return best

def _metropolis_mc(centers, time_budget=12.0):
    """Metropolis Monte Carlo to escape local minima."""
    start = time.time()
    best = centers.copy()
    best_r = _compute_radius(centers)
    current = centers.copy()
    current_r = best_r
    
    T = 0.01
    step_size = 0.02
    rng = np.random.RandomState(12345)
    
    for it in range(3000):
        if time.time() - start > time_budget:
            break
        
        T *= 0.999
        step_size *= 0.9995
        
        i = rng.randint(N)
        delta = rng.normal(0, step_size, size=3)
        new_pos = current[i] + delta
        new_pos = np.clip(new_pos, -0.49, 0.49)
        
        trial = current.copy()
        trial[i] = new_pos
        trial_r = _compute_radius(trial)
        
        delta_r = trial_r - current_r
        if delta_r > 0 or rng.random() < np.exp(delta_r / T):
            current = trial
            current_r = trial_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
    
    return best

def _local_search(centers, time_budget=8.0):
    """Final local search with Nelder-Mead."""
    start = time.time()
    try:
        def neg_radius(c):
            c = c.reshape(N, 3)
            return -_compute_radius(c)
        
        res = minimize(
            neg_radius,
            centers.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if res.x is not None and np.all(np.isfinite(res.x)):
            return res.x.reshape(N, 3)
    except Exception:
        pass
    return centers

def _seed_fcc():
    """FCC-like lattice initialization."""
    pts = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                x = (i + 0.5) / 3.0 - 0.5
                y = (j + 0.5) / 3.0 - 0.5
                z = (k + 0.5) / 3.0 - 0.5
                pts.append([x, y, z])
                pts.append([x, y + 1/6, z + 1/6])
                pts.append([x + 1/6, y, z + 1/6])
                pts.append([x + 1/6, y + 1/6, z])
    
    pts = np.array(pts)
    mask = np.all(np.abs(pts) <= 0.49, axis=1)
    pts = pts[mask]
    
    if len(pts) < N:
        rng = np.random.RandomState(42)
        extra = rng.uniform(-0.4, 0.4, size=(N - len(pts), 3))
        pts = np.vstack([pts, extra])
    
    return pts[:N].copy()

def _seed_grid():
    """Grid initialization."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def construct():
    start_time = time.time()
    time_limit = 140.0
    
    best_centers = None
    best_radius = -np.inf
    
    def _evaluate(centers):
        nonlocal best_centers, best_radius
        if not np.all(np.isfinite(centers)):
            return
        if np.any(np.abs(centers) > 0.5):
            return
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    seeds = [_seed_grid(), _seed_fcc()]
    
    for seed in seeds:
        if time.time() - start_time > time_limit - 40:
            break
        
        current = seed.copy()
        _evaluate(current)
        
        if time.time() - start_time < time_limit - 35:
            current = _lloyd_optimize(current, time_budget=12.0)
            _evaluate(current)
        
        if time.time() - start_time < time_limit - 28:
            current = _repulsion_anneal(current, time_budget=15.0)
            _evaluate(current)
        
        if time.time() - start_time < time_limit - 20:
            current = _metropolis_mc(current, time_budget=12.0)
            _evaluate(current)
        
        if time.time() - start_time < time_limit - 12:
            for alpha in [16.0, 64.0]:
                if time.time() - start_time > time_limit - 8:
                    break
                current = _smooth_polish(current, alpha, time_budget=6.0)
                _evaluate(current)
        
        if time.time() - start_time < time_limit - 5:
            current = _local_search(current, time_budget=8.0)
            _evaluate(current)
    
    if best_centers is None:
        return _seed_grid()
    
    best_centers = np.clip(best_centers, -0.5 + 1e-15, 0.5 - 1e-15)
    return best_centers
