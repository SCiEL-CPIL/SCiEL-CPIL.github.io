"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
import time
import sys

N = 200
DEADLINE = 140.0  # Leave 10s margin for the 150s limit

def _score(pts):
    """Compute the raw score (largest common radius) for a set of points in [-0.5, 0.5]^2."""
    # Boundary check
    min_bound = 0.5 - np.max(np.abs(pts), axis=1)
    if np.any(min_bound <= 0):
        return -1e-9
    min_dist = np.min(cKDTree(pts).query(pts, k=2)[0][:, 1])
    return min(0.5 * min_dist, min_bound)

def _gradient(pts, r_target):
    """
    Approximate gradient of the score with respect to points.
    We use a simple subgradient approach: push points that are too close or too close to boundary.
    """
    grads = np.zeros_like(pts)
    # Boundary gradients
    # For each point, if it's close to boundary, push it inward
    # The constraint is 0.5 - max(|x|, |y|) >= r
    # Gradient of -max(|x|,|y|) is -sign of the dominant coordinate
    
    # Pairwise gradients
    # We want to increase the minimum distance.
    # For each pair (i,j), if dist(i,j) is the minimum, push i and j apart.
    # To be efficient, we only consider the closest neighbors.
    
    # Actually, let's use a simpler approach:
    # The objective is min(0.5 * d_min, b_min)
    # Let's just maximize the minimum of (0.5 * d_min, b_min).
    # This is equivalent to maximizing r such that all constraints are satisfied.
    
    # For a simple gradient step:
    # 1. If boundary is the bottleneck, push points away from boundary.
    # 2. If pairwise distance is the bottleneck, push close pairs apart.
    
    b_vals = 0.5 - np.max(np.abs(pts), axis=1)
    # Find closest pairs
    tree = cKDTree(pts)
    pairs, dists = tree.query(pts, k=2)
    min_d = dists[:, 1]
    
    # Determine which constraint is active
    active_boundary = np.where(b_vals == np.min(b_vals))[0]
    active_pair_i = np.argmin(min_d)
    
    # Gradient for boundary
    if np.min(b_vals) <= 0.5 * min_d[active_pair_i]:
        # Boundary is bottleneck
        for i in active_boundary:
            x, y = pts[i]
            if abs(x) >= abs(y):
                grads[i, 0] = -np.sign(x)
            else:
                grads[i, 1] = -np.sign(y)
    else:
        # Pairwise is bottleneck
        i = active_pair_i
        j = pairs[i, 1]
        diff = pts[i] - pts[j]
        norm = np.linalg.norm(diff)
        if norm > 1e-12:
            grads[i] += diff / norm
            grads[j] -= diff / norm
    
    return grads

def construct():
    start_time = time.time()
    
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts = grid[:N].copy()
    
    best_score = _score(pts)
    best_pts = pts.copy()
    
    # Local optimization using gradient descent with random restarts
    # We'll do a few iterations of gradient ascent on the score
    
    n_iter = 500
    lr = 0.001
    
    for it in range(n_iter):
        if time.time() - start_time > DEADLINE:
            break
        
        # Compute gradient
        grad = _gradient(pts, best_score)
        
        # Update
        pts_new = pts + lr * grad
        
        # Project back into square
        pts_new = np.clip(pts_new, -0.49, 0.49)
        
        new_score = _score(pts_new)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = pts_new.copy()
            pts = pts_new.copy()
            # Decrease learning rate slowly
            lr *= 0.999
        else:
            # Try a smaller step
            pts_trial = pts + lr * 0.1 * grad
            pts_trial = np.clip(pts_trial, -0.49, 0.49)
            trial_score = _score(pts_trial)
            if trial_score > best_score:
                best_score = trial_score
                best_pts = pts_trial.copy()
                pts = pts_trial.copy()
            else:
                # Random perturbation
                pts_trial = pts + 0.005 * np.random.randn(N, 2)
                pts_trial = np.clip(pts_trial, -0.49, 0.49)
                trial_score = _score(pts_trial)
                if trial_score > best_score:
                    best_score = trial_score
                    best_pts = pts_trial.copy()
                    pts = pts_trial.copy()
    
    # Final polish: try to maximize radius via binary search on r
    # Check if a given radius r is feasible
    def is_feasible(r):
        # All points must be at least r from boundary
        # And at least 2r from each other
        # This is a constraint satisfaction problem, but for now we just check current points
        b_vals = 0.5 - np.max(np.abs(best_pts), axis=1)
        if np.any(b_vals < r):
            return False
        min_d = np.min(cKDTree(best_pts).query(best_pts, k=2)[0][:, 1])
        return 0.5 * min_d >= r
    
    # Binary search on r to find the maximum feasible radius for the current configuration
    # But this doesn't improve the configuration, it just measures it.
    # So we'll just return the best configuration we found.
    
    return best_pts
# EVOLVE-BLOCK-END
