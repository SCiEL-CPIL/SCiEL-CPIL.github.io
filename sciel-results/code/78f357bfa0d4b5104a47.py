"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_score(pts):
    """Compute the packing score: min of pairwise distances/2 and (1 - |c_i|)."""
    if len(pts) < N:
        return 0.0
    dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    min_boundary = 1.0 - np.max(np.linalg.norm(pts, axis=1))
    return min(min_pair_dist / 2.0, min_boundary)

def _hex_lattice_init(n=N):
    """Generate initial points from a hexagonal lattice cut to the unit disk."""
    # Estimate spacing for n points in unit disk
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= n:
            # Sort by distance from center to keep most central points
            idx = np.argsort(radius, kind="stable")[:n]
            return points[idx]
        spacing *= 0.98

def _slsqp_optimize(pts, max_iter=500, tol=1e-8):
    """Use SLSQP to maximize the minimum radius."""
    n = len(pts)
    
    def objective(x):
        """Negative of the score (we minimize -score)."""
        p = x.reshape(n, 2)
        # Check boundary
        norms = np.linalg.norm(p, axis=1)
        if np.any(norms > 1.0):
            return 1.0  # Large penalty
        # Pairwise distances
        dists = np.linalg.norm(p[:, np.newaxis, :] - p[np.newaxis, :, :], axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        min_bound = 1.0 - np.max(norms)
        score = min(min_pair / 2.0, min_bound)
        return -score
    
    # Constraints: each point must be inside unit circle
    constraints = []
    for i in range(n):
        def make_constraint(i):
            def constraint(x):
                p = x.reshape(n, 2)
                return 1.0 - np.linalg.norm(p[i])
            return constraint
        constraints.append({'type': 'ineq', 'fun': make_constraint(i)})
    
    # Bounds: each coordinate in [-1, 1]
    bounds = [(-1.0, 1.0)] * (2 * n)
    
    result = minimize(objective, pts.flatten(), method='SLSQP', 
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': max_iter, 'ftol': tol, 'disp': False})
    
    if result.success:
        return result.x.reshape(n, 2)
    return pts

def _local_polish(pts, steps=100, lr=0.01):
    """Local polish using gradient ascent on the score."""
    n = len(pts)
    p = pts.copy()
    best_score = _compute_score(p)
    best_p = p.copy()
    
    for step in range(steps):
        # Compute score and gradients numerically
        score = _compute_score(p)
        if score > best_score:
            best_score = score
            best_p = p.copy()
        
        # Compute gradient via finite differences
        eps = 1e-6
        grad = np.zeros_like(p)
        for i in range(n):
            for j in range(2):
                p_old = p[i, j]
                p[i, j] = p_old + eps
                s_plus = _compute_score(p)
                p[i, j] = p_old - eps
                s_minus = _compute_score(p)
                p[i, j] = p_old
                grad[i, j] = (s_plus - s_minus) / (2 * eps)
        
        # Update
        p = p + lr * grad
        # Project back to valid region
        norms = np.linalg.norm(p, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            p[mask] = p[mask] / norms[mask, np.newaxis] * 0.999
    
    return best_p

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Step 1: Initialize with hexagonal lattice
    pts = _hex_lattice_init(N)
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    # Step 2: SLSQP optimization
    if time.time() < deadline - 20.0:
        try:
            optimized = _slsqp_optimize(pts, max_iter=300)
            score = _compute_score(optimized)
            if score > best_score:
                best_score = score
                best_pts = optimized
        except Exception:
            pass
    
    # Step 3: Local polish
    if time.time() < deadline - 5.0:
        try:
            polished = _local_polish(best_pts, steps=50, lr=0.005)
            score = _compute_score(polished)
            if score > best_score:
                best_score = score
                best_pts = polished
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
