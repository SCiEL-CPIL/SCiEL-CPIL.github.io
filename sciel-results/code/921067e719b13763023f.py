"""Improved packing: local search with simulated annealing + L-BFGS-B polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

# Best known radius ~0.071377103865
# Square grid baseline: m=8, spacing=1/8, r=1/16=0.0625


def _compute_radius(pts):
    """Compute the achievable radius for a set of points."""
    # Distance to walls
    wall_dists = np.minimum(0.5 - np.abs(pts), axis=1)
    min_wall = np.min(wall_dists)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    return min(min_wall, min_pair / 2.0)


def _score(pts):
    """Score function: larger is better."""
    r = _compute_radius(pts)
    return r


def _init_grid():
    """Initialize with a square grid (8x8, take first 50)."""
    m = 8
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _init_random(seed=0):
    """Initialize with random points, then do a quick local optimization."""
    rng = np.random.RandomState(seed)
    pts = rng.uniform(-0.5, 0.5, size=(N, 2))
    # Quick local optimization
    for _ in range(50):
        for i in range(N):
            # Move point i to maximize min distance
            best_pt = pts[i].copy()
            best_r = -1
            for trial in range(20):
                cand = pts[i] + rng.uniform(-0.02, 0.02, size=2)
                cand = np.clip(cand, -0.49, 0.49)
                pts[i] = cand
                r = _score(pts)
                if r > best_r:
                    best_r = r
                    best_pt = cand.copy()
                else:
                    pts[i] = best_pt
    return pts.copy()


def _local_search(pts, time_limit, rng):
    """Simulated annealing local search."""
    best_pts = pts.copy()
    best_score = _score(pts)
    current_pts = pts.copy()
    current_score = best_score
    
    start_time = time.time()
    temp = 0.01
    cooling = 0.995
    step_size = 0.05
    
    while time.time() - start_time < time_limit:
        if temp < 1e-8:
            break
        
        # Pick a random point to move
        i = rng.randint(N)
        old_pt = current_pts[i].copy()
        
        # Random perturbation
        perturbation = rng.normal(0, step_size, size=2)
        current_pts[i] = np.clip(current_pts[i] + perturbation, -0.49, 0.49)
        
        new_score = _score(current_pts)
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / temp):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = current_pts.copy()
        else:
            current_pts[i] = old_pt
        
        temp *= cooling
    
    return best_pts, best_score


def _polish_lbfgsb(pts):
    """Polish using L-BFGS-B to maximize the minimum radius."""
    def neg_radius(pts_flat):
        pts = pts_flat.reshape(-1, 2)
        return -_compute_radius(pts)
    
    def neg_radius_grad(pts_flat):
        pts = pts_flat.reshape(-1, 2)
        # Compute gradient numerically
        eps = 1e-7
        grad = np.zeros_like(pts_flat)
        base = neg_radius(pts_flat)
        for j in range(len(pts_flat)):
            pts_flat[j] += eps
            f_plus = neg_radius(pts_flat)
            pts_flat[j] -= 2 * eps
            f_minus = neg_radius(pts_flat)
            pts_flat[j] += eps
            grad[j] = (f_plus - f_minus) / (2 * eps)
        return grad
    
    # Use bounds to keep points inside the square
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    result = minimize(
        neg_radius,
        pts.flatten(),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    polished_pts = result.x.reshape(-1, 2)
    polished_score = _score(polished_pts)
    
    return polished_pts, polished_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Start with grid initialization
    rng = np.random.RandomState(42)
    
    # Try multiple initializations
    candidates = []
    
    # 1. Square grid
    grid_pts = _init_grid()
    grid_score = _score(grid_pts)
    candidates.append((grid_pts, grid_score))
    
    # 2. Random initializations with local search
    for seed in range(10):
        if time.time() > deadline - 60:
            break
        rand_pts = _init_random(seed)
        rand_score = _score(rand_pts)
        candidates.append((rand_pts, rand_score))
    
    # Sort by score, keep top few
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = candidates[:5]
    
    # Local search on top candidates
    best_pts = None
    best_score = -1
    
    for pts, score in top_candidates:
        if time.time() > deadline - 30:
            break
        ls_pts, ls_score = _local_search(pts, time_limit=20, rng=rng)
        if ls_score > best_score:
            best_score = ls_score
            best_pts = ls_pts
    
    # Final polish with L-BFGS-B
    if best_pts is not None and time.time() < deadline - 10:
        polished_pts, polished_score = _polish_lbfgsb(best_pts)
        if polished_score > best_score:
            best_pts = polished_pts
            best_score = polished_score
    
    # Safety: if no good result, return grid
    if best_pts is None:
        return _init_grid()
    
    # Ensure points are strictly inside
    best_pts = np.clip(best_pts, -0.4999, 0.4999)
    
    return best_pts
# EVOLVE-BLOCK-END
