"""
Strategy: 2-Level Optimization with MDS Embedding
1. Optimize the set of 28 unique pairwise distances to maximize their product.
2. Embed these distances into 2D using MDS (classical) or iterative coordinate descent.
3. Refine the coordinates directly to maximize the score.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin

def _log_score(points):
    """Compute log of Delta/n^n."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    mask = ~np.eye(n, dtype=bool)
    valid_dists = dists[mask]
    if np.any(valid_dists <= 0):
        return -np.inf
    diameter = np.max(dists)
    if diameter <= 0:
        return -np.inf
    n_pairs = n * (n - 1)
    # Delta = prod(d_ij) / diameter^(n_pairs) * 2^(n_pairs)
    # Score = Delta / n^n
    log_delta = n_pairs * np.log(2.0) + np.sum(np.log(valid_dists)) - n_pairs * np.log(diameter)
    log_score = log_delta - n * np.log(n)
    return log_score

def _score(points):
    ls = _log_score(points)
    if not np.isfinite(ls):
        return -np.inf
    if ls > 50:
        return float('inf')
    return math.exp(ls)

def _embed_distances(dists_1d, init_points=None):
    """
    Embed 1D array of pairwise distances (pdist format) into 2D points.
    Uses iterative refinement to minimize stress (difference between target and actual distances).
    """
    n = 8
    if init_points is None:
        # Initialize with random points
        init_points = np.random.randn(n, 2) * 0.5
    
    pts = init_points.copy()
    
    # Target distance matrix
    target_dist_mat = squareform(dists_1d)
    
    # Simple gradient descent on coordinates to match target distances
    # Objective: sum((d_actual - d_target)^2)
    for it in range(200):
        if time.time() > _DEADLINE - 10:
            break
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        actual_dists = np.sqrt(np.sum(diff**2, axis=2))
        # Avoid division by zero
        actual_dists[actual_dists < 1e-8] = 1e-8
        
        # Gradient of stress w.r.t. coordinates
        # stress = sum_{i<j} (d_ij - t_ij)^2
        # d_stress/d_pt_i = -4 * sum_{j!=i} (d_ij - t_ij) * (pt_i - pt_j) / d_ij
        
        grad = np.zeros_like(pts)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d_ij = actual_dists[i, j]
                t_ij = target_dist_mat[i, j]
                grad[i] -= 4.0 * (d_ij - t_ij) * (pts[i] - pts[j]) / d_ij
        
        # Update
        pts -= 0.01 * grad
        
    return pts

def _optimize_distances(d0, max_iter=500):
    """
    Optimize the 28 pairwise distances to maximize the product.
    Constraint: distances must be embeddable in 2D (handled by embedding step).
    We optimize log(dists) to keep them positive.
    """
    def neg_log_prod(log_dists):
        if time.time() > _DEADLINE - 5:
            return 1e10
        dists = np.exp(log_dists)
        
        # Check if embeddable: try to embed and check stress
        # If stress is too high, penalize
        try:
            pts = _embed_distances(dists, init_points=None)
            actual_dists = pdist(pts)
            stress = np.sum((actual_dists - dists)**2)
            # If stress is high, the configuration is not geometrically valid
            if stress > 0.1:
                return 1e10 + stress
        except Exception:
            return 1e10
        
        # Score based on product of distances
        # We want to maximize product, so minimize -log(prod)
        # But we also need to scale to diameter 2.
        # The score function handles scaling.
        # Here, we just maximize the raw product of distances.
        return -np.sum(log_dists)
    
    res = minimize(neg_log_prod, d0, method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6})
    return res.x

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 145.0
    
    best_points = None
    best_score = -np.inf
    best_log_score = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Start with regular octagon
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    oct_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    best_points = oct_pts
    best_score = _score(oct_pts)
    best_log_score = _log_score(oct_pts)
    
    # Strategy: Generate diverse distance profiles, embed, and refine.
    
    # 1. Regular Octagon Distances
    oct_dists = pdist(oct_pts)
    
    # 2. Two Concentric Squares Distances
    # Try various radii and rotations
    candidate_dists = [oct_dists]
    
    for r1 in [0.5, 0.7, 0.9, 1.1]:
        for r2 in [0.5, 0.7, 0.9, 1.1]:
            if abs(r1 - r2) < 0.1:
                continue
            for rot1 in [0, np.pi/4]:
                for rot2 in [0, np.pi/4]:
                    pts = []
                    for k in range(4):
                        a = rot1 + k * np.pi / 2
                        pts.append([r1 * np.cos(a), r1 * np.sin(a)])
                    for k in range(4):
                        a = rot2 + k * np.pi / 2
                        pts.append([r2 * np.cos(a), r2 * np.sin(a)])
                    pts = np.array(pts)
                    candidate_dists.append(pdist(pts))
                    
    # 3. Random Distance Profiles
    for _ in range(20):
        # Random distances between 0.5 and 2.0
        random_dists = rng.uniform(0.5, 2.0, 28)
        candidate_dists.append(random_dists)
        
    # Process each candidate distance profile
    for dists in candidate_dists:
        if time.time() > _DEADLINE - 20:
            break
        
        # Optimize distances
        try:
            opt_log_dists = _optimize_distances(np.log(dists), max_iter=200)
            opt_dists = np.exp(opt_log_dists)
        except Exception:
            continue
            
        # Embed the optimized distances
        try:
            # Use multiple random initializations for embedding
            best_embed_pts = None
            best_embed_stress = np.inf
            
            for _ in range(5):
                init = rng.randn(8, 2) * 0.5
                pts = _embed_distances(opt_dists, init_points=init)
                actual_dists = pdist(pts)
                stress = np.sum((actual_dists - opt_dists)**2)
                if stress < best_embed_stress:
                    best_embed_stress = stress
                    best_embed_pts = pts
                    
            if best_embed_pts is None or best_embed_stress > 0.5:
                continue
                
            # Refine the embedded points directly to maximize score
            # Use Nelder-Mead on the coordinates
            def neg_log_score_coords(params):
                if time.time() > _DEADLINE - 5:
                    return 1e10
                pts = params.reshape(8, 2)
                ls = _log_score(pts)
                if not np.isfinite(ls):
                    return 1e10
                return -ls
            
            res = minimize(neg_log_score_coords, best_embed_pts.flatten(),
                           method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res.fun < 1e10:
                refined_pts = res.x.reshape(8, 2)
                ls_val = -res.fun
                if np.isfinite(ls_val) and ls_val > best_log_score:
                    best_log_score = ls_val
                    best_score = math.exp(ls_val) if ls_val < 50 else float('inf')
                    best_points = refined_pts
                    
        except Exception:
            continue
        
    # Final polish on the best points found
    if time.time() < _DEADLINE - 10 and best_points is not None:
        def neg_log_score_coords(params):
            if time.time() > _DEADLINE - 5:
                return 1e10
            pts = params.reshape(8, 2)
            ls = _log_score(pts)
            if not np.isfinite(ls):
                return 1e10
            return -ls
        
        # L-BFGS-B with smooth approximation if possible, or just NM
        try:
            res = minimize(neg_log_score_coords, best_points.flatten(),
                           method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < 1e10:
                refined_pts = res.x.reshape(8, 2)
                ls_val = -res.fun
                if np.isfinite(ls_val) and ls_val > best_log_score:
                    best_log_score = ls_val
                    best_score = math.exp(ls_val) if ls_val < 50 else float('inf')
                    best_points = refined_pts
        except Exception:
            pass

    # Ensure coordinates are within bounds
    final_result = []
    for p in best_points:
        x = max(-1e6, min(1e6, float(p[0])))
        y = max(-1e6, min(1e6, float(p[1])))
        final_result.append([x, y])

    return final_result
