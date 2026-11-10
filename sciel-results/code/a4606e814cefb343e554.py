# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    # SDP-inspired approach: optimize Gram matrix G (14x14) with G_ii=1, G PSD
    # d_ij^2 = 2 - 2*G_ij
    # Maximize min(d_ij^2) subject to max(d_ij^2) <= 1
    # i.e., maximize t subject to 2-2*G_ij >= t and 2-2*G_ij <= 1 for all i<j
    # The latter means G_ij >= 0.5
    
    # Since we can't solve SDP directly, use a two-stage approach:
    # 1. Use DE to find good points in coordinate space
    # 2. Refine with multi-start Nelder-Mead
    
    def neg_ratio(flat_points):
        pts = flat_points.reshape(n, d)
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-15:
            return 0.0
        return -(d_min / d_max)
    
    # Stage 1: Differential evolution with penalty
    def objective(flat_points):
        points = flat_points.reshape(n, d)
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-15:
            return 1.0
        # We want to maximize d_min/d_max
        # Penalize if d_max > 1 (shouldn't happen much)
        penalty = 100.0 * np.maximum(0.0, d_max - 1.0)
        return -(d_min / d_max) + penalty
    
    bounds = [(-1.0, 1.0)] * (n * d)
    
    res = differential_evolution(
        objective, 
        bounds, 
        seed=42, 
        maxiter=500, 
        popsize=30,
        tol=1e-12,
        polish=True
    )
    
    best_points = res.x.reshape(n, d)
    best_score = -neg_ratio(best_points.flatten())
    
    # Stage 2: Multi-start Nelder-Mead refinement
    rng = np.random.RandomState(42)
    
    # Generate multiple starting points: DE result + random perturbations
    starts = [best_points.copy()]
    for _ in range(50):
        # Random points on sphere of radius ~0.5
        pts = rng.randn(n, d) * 0.5
        # Normalize to have max distance ~1
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max > 1e-10:
            pts = pts / d_max
        starts.append(pts)
    
    # Also try known good configurations
    # Icosahedron vertices (12 points) + 2 more
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    # Add 2 more points to make 14
    extra = rng.randn(2, d)
    extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
    ico14 = np.vstack([ico, extra])
    dists = pdist(ico14)
    d_max = np.max(dists)
    if d_max > 1e-10:
        ico14 = ico14 / d_max
    starts.append(ico14)
    
    for start in starts:
        if not np.all(np.isfinite(start)):
            continue
        res_nm = minimize(
            neg_ratio, 
            start.flatten(), 
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16}
        )
        if res_nm.success or res_nm.fun < neg_ratio(start.flatten()):
            cand = res_nm.x.reshape(n, d)
            if np.all(np.isfinite(cand)):
                score = -neg_ratio(cand.flatten())
                if score > best_score:
                    best_score = score
                    best_points = cand
    
    points = best_points
    
    # Final scale to ensure max distance is exactly 1
    dists = pdist(points)
    d_max = np.max(dists)
    if d_max > 1e-10:
        points = points / d_max
    
    # Ensure finite and correct shape
    if not np.all(np.isfinite(points)):
        np.random.seed(42)
        points = np.random.randn(n, d)
    
    return points

# EVOLVE-BLOCK-END
