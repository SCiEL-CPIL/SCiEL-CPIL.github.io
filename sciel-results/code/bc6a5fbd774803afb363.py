"""Pack 50 equal spheres in a unit cube using simulated annealing / local search."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, cdist


def _compute_radius(pts):
    """Compute the maximum common radius for given centers."""
    # Distance to walls
    wall_dists = 0.5 - np.abs(pts)
    min_wall = wall_dists.min()
    # Pairwise distances
    if len(pts) > 1:
        d = pdist(pts)
        min_pair = d.min()
    else:
        min_pair = np.inf
    return min(min_wall, min_pair / 2.0)


def _initial_grid(n):
    """Generate an initial grid-like arrangement."""
    m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n].astype(np.float64)


def _initial_random(n, seed=42):
    """Generate random initial positions."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))


def _local_search(pts, max_iter=500, step_size=0.01, cooling=0.999, seed=0):
    """Perform a local search with simulated annealing."""
    rng = np.random.default_rng(seed)
    best = pts.copy()
    best_r = _compute_radius(best)
    current = pts.copy()
    current_r = best_r
    T = 0.01
    
    for it in range(max_iter):
        # Pick a random point and a random direction
        idx = rng.integers(0, len(current))
        direction = rng.normal(size=3)
        direction /= (np.linalg.norm(direction) + 1e-12)
        
        # Try a small move
        delta = step_size * (0.5 + 0.5 * rng.random()) * direction
        new_pt = current[idx].copy()
        new_pt += delta
        # Clamp to cube
        new_pt = np.clip(new_pt, -0.499, 0.499)
        
        old_pt = current[idx].copy()
        current[idx] = new_pt
        new_r = _compute_radius(current)
        
        # Accept if better, or with some probability if worse
        if new_r >= current_r or rng.random() < np.exp((new_r - current_r) / max(T, 1e-8)):
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = current.copy()
        else:
            current[idx] = old_pt
        
        T *= cooling
        if T < 1e-10:
            break
    
    return best, best_r


def _gradient_ascent(pts, max_iter=200, lr=0.005):
    """Gradient ascent on the minimum distance."""
    current = pts.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    for it in range(max_iter):
        # Compute the limiting constraint
        # Find which pair or wall gives the minimum
        wall_dists = 0.5 - np.abs(current)
        min_wall_idx = np.argmin(wall_dists.min(axis=1))
        min_wall_val = wall_dists[min_wall_idx].min()
        
        d = pdist(current)
        min_pair_idx = np.argmin(d)
        # Convert flat index to pair indices
        n = len(current)
        i = int((min_pair_idx - (n - 1)) // 2)
        j = min_pair_idx - i * (n - 1) - i
        min_pair_val = d[min_pair_idx] / 2.0
        
        if min_wall_val < min_pair_val:
            # Move the point away from the wall
            p = current[min_wall_idx]
            # Find which coordinate is limiting
            coord = np.argmin(np.abs(p))
            if p[coord] > 0:
                grad = np.zeros(3)
                grad[coord] = -1.0
            else:
                grad = np.zeros(3)
                grad[coord] = 1.0
            current[min_wall_idx] += lr * grad
        else:
            # Move the two points apart
            p1 = current[i]
            p2 = current[j]
            diff = p1 - p2
            norm = np.linalg.norm(diff)
            if norm > 1e-12:
                grad = diff / norm
                current[i] += lr * grad
                current[j] -= lr * grad
        
        # Clamp to cube
        current = np.clip(current, -0.499, 0.499)
        
        new_r = _compute_radius(current)
        if new_r > best_r:
            best_r = new_r
            best = current.copy()
        
        # Reduce learning rate
        lr *= 0.995
    
    return best, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Start with a good initial configuration
    pts = _initial_grid(N)
    best = pts.copy()
    best_r = _compute_radius(best)
    
    # Try a few different initializations and local searches
    seeds = [42, 123, 456, 789, 1000]
    for seed in seeds:
        if time.time() > deadline:
            break
        
        # Random restart
        pts_rand = _initial_random(N, seed=seed)
        r_rand = _compute_radius(pts_rand)
        
        # Local search from random start
        pts_ls, r_ls = _local_search(pts_rand, max_iter=300, step_size=0.02, seed=seed)
        if r_ls > best_r:
            best_r = r_ls
            best = pts_ls.copy()
        
        # Gradient ascent from best
        pts_grad, r_grad = _gradient_ascent(best, max_iter=100, lr=0.01)
        if r_grad > best_r:
            best_r = r_grad
            best = pts_grad.copy()
        
        # More local search from the improved point
        pts_ls2, r_ls2 = _local_search(best, max_iter=400, step_size=0.01, seed=seed + 1000)
        if r_ls2 > best_r:
            best_r = r_ls2
            best = pts_ls2.copy()
    
    # Final refinement
    if time.time() < deadline - 5.0:
        pts_final, r_final = _gradient_ascent(best, max_iter=200, lr=0.005)
        if r_final > best_r:
            best_r = r_final
            best = pts_final.copy()
        
        pts_final2, r_final2 = _local_search(best, max_iter=300, step_size=0.005, seed=9999)
        if r_final2 > best_r:
            best_r = r_final2
            best = pts_final2.copy()
    
    # Ensure validity
    best = np.clip(best, -0.499, 0.499)
    
    return best
# EVOLVE-BLOCK-END
