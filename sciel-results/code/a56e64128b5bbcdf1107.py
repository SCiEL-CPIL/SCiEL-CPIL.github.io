N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius_fast(pts):
    """Compute the maximum packing radius for given points."""
    if len(pts) < 2:
        return 0.0
    
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    radii = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(1.0 - radii)
    
    return min(min_pair_dist / 2.0, min_boundary)

def _init_hex(N):
    """Initialize points using a hexagonal lattice."""
    r_est = np.sqrt(0.9069 / N)
    spacing = 2 * r_est * 1.05
    
    k = int(1.0 / spacing) + 5
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0:
                points.append((x, y))
    
    points = np.array(points)
    if len(points) < N:
        return np.random.uniform(-1, 1, (N, 2))
    
    dists_center = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(dists_center, kind='stable')[:N]
    return points[idx]

def _hertzian_dynamics(pts, iters=2000, deadline=None):
    """Force-directed relaxation."""
    pts = pts.copy()
    N = len(pts)
    r = _compute_radius_fast(pts)
    
    for it in range(iters):
        if deadline and time.time() > deadline:
            break
            
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        threshold = 2.2 * r
        mask = dists < threshold
        
        dists_sq = dists ** 2
        np.fill_diagonal(dists_sq, 1.0)
        weights = np.where(mask, 1.0 / dists_sq, 0.0)
        
        forces = np.sum(diff * weights[:, :, np.newaxis], axis=1)
        
        radii = np.hypot(pts[:, 0], pts[:, 1])
        dist_to_bound = 1.0 - radii
        bound_mask = dist_to_bound < r
        norms = np.maximum(radii, 1e-10)
        inwards = -pts / norms[:, np.newaxis]
        bound_weights = np.where(bound_mask, 1.0 / np.maximum(dist_to_bound, 1e-6), 0.0)
        forces += inwards * bound_weights[:, np.newaxis]
        
        step = 0.1 * r
        forces_norm = np.hypot(forces[:, 0], forces[:, 1])
        max_force = np.max(forces_norm)
        if max_force > 0:
            scale = step / max_force
            pts += forces * scale
        
        radii = np.hypot(pts[:, 0], pts[:, 1])
        outside = radii > 1.0
        if np.any(outside):
            pts[outside] = pts[outside] / radii[outside, np.newaxis]
            
        new_r = _compute_radius_fast(pts)
        if new_r > r:
            r = new_r
            
    return pts

def _polish(pts, deadline):
    """Local search optimization."""
    pts = pts.copy()
    N = len(pts)
    r = _compute_radius_fast(pts)
    
    for pass_num in range(20):
        if deadline and time.time() > deadline:
            break
            
        improved = False
        for i in range(N):
            if deadline and time.time() > deadline - 5:
                break
                
            p_i = pts[i]
            
            def obj(x_y):
                p_new = np.array(x_y)
                d_bound = 1.0 - np.hypot(p_new[0], p_new[1])
                if d_bound < 0:
                    return -1000 + d_bound
                
                diffs = pts - p_new
                dists = np.hypot(diffs[:, 0], diffs[:, 1])
                dists[i] = np.inf
                min_pair = np.min(dists)
                
                return -min(min_pair / 2.0, d_bound)
            
            try:
                res = minimize(
                    obj, 
                    p_i, 
                    method='L-BFGS-B', 
                    bounds=[(-1.0, 1.0), (-1.0, 1.0)],
                    options={'maxiter': 20, 'ftol': 1e-6}
                )
                if res.fun < obj(p_i) - 1e-8:
                    pts[i] = res.x
                    improved = True
            except:
                pass
        
        new_r = _compute_radius_fast(pts)
        if new_r > r:
            r = new_r
        else:
            if not improved:
                break

    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    pts = _init_hex(N)
    pts = _hertzian_dynamics(pts, iters=2000, deadline=deadline - 20.0)
    pts = _polish(pts, deadline=deadline)
    
    if pts.shape != (N, 2):
        pts = np.zeros((N, 2))
        
    return pts
# EVOLVE-BLOCK-END
