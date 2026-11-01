"""Improved packing: local optimization from grid init with active constraint repulsion."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 50

def _score(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    # Wall constraint: min over i,k of (0.5 - |c_ik|)
    wall_min = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_min)
    
    # Pairwise distances
    n = len(centers)
    if n < 2:
        return r_wall
    
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Get minimum pairwise distance (excluding diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    
    return min(r_wall, r_pair)

def construct():
    t_start = time.time()
    deadline = t_start + 140  # 140s safety margin before 150s limit
    
    # Initial grid placement
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    best_centers = centers.copy()
    best_score = _score(best_centers)
    
    # Local optimization using L-BFGS-B
    def objective(x):
        """Negative score to maximize."""
        c = x.reshape(-1, 3)
        # Add small penalty for going outside bounds
        out_of_bounds = np.maximum(0, np.abs(c) - 0.4999)
        penalty = np.sum(out_of_bounds ** 2)
        s = _score(c)
        return -s + penalty
    
    # Try optimization
    try:
        res = minimize(objective, centers.ravel(), method='L-BFGS-B', 
                      bounds=[(-0.5, 0.5)] * 3 * N,
                      options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.fun < -best_score:  # Improved
            new_centers = res.x.reshape(-1, 3)
            new_score = _score(new_centers)
            if new_score > best_score:
                best_centers = new_centers
                best_score = new_score
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return best_centers
    
    # Second pass: more iterations
    try:
        res = minimize(objective, best_centers.ravel(), method='L-BFGS-B',
                      bounds=[(-0.5, 0.5)] * 3 * N,
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if res.fun < -best_score:
            new_centers = res.x.reshape(-1, 3)
            new_score = _score(new_centers)
            if new_score > best_score:
                best_centers = new_centers
                best_score = new_score
    except Exception:
        pass
    
    # Check time again
    if time.time() > deadline:
        return best_centers
    
    # Final polish: coordinate descent
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(N):
            # Try small perturbations in each coordinate
            for coord in range(3):
                for delta in [0.001, -0.001, 0.0005, -0.0005, 0.002, -0.002]:
                    new_c = best_centers.copy()
                    new_c[i, coord] += delta
                    if np.all(np.abs(new_c) <= 0.5):
                        s = _score(new_c)
                        if s > best_score:
                            best_centers = new_c
                            best_score = s
                            improved = True
    
    return best_centers
