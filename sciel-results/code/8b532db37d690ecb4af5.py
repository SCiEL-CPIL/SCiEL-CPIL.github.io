"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_score(pts):
    """Compute the valid packing radius for given center positions."""
    n = len(pts)
    if n == 0:
        return 0.0
    # Distance to container boundary
    dist_to_boundary = 1.0 - np.linalg.norm(pts, axis=1)
    min_boundary = np.min(dist_to_boundary)
    
    # Minimum pairwise distance
    if n > 1:
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Set diagonal to infinity to exclude self-pairs
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
    else:
        min_pair = np.inf
    
    return min(min_boundary, min_pair / 2.0)


def _hex_init(n_points, n_restarts=5, seed=42):
    """Generate multiple hexagonal lattice initializations with different parameters."""
    rng = np.random.RandomState(seed)
    candidates = []
    
    for restart in range(n_restarts):
        # Vary the spacing slightly for diversity
        base_spacing = 2.0 / (np.sqrt(n_points) + 1.0)
        spacing = base_spacing * (0.95 + 0.1 * rng.random())
        
        k = int(1.0 / spacing) + 3
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        
        points = np.array(rows)
        radii = np.hypot(points[:, 0], points[:, 1])
        
        # Filter points that fit in the unit circle with margin
        margin = spacing / 2.0
        mask = radii <= 1.0 - margin
        points = points[mask]
        
        if len(points) >= n_points:
            # Sort by distance from center, take closest n_points
            order = np.argsort(radii[mask], kind="stable")[:n_points]
            pts = points[order].copy()
            
            # Add small random perturbation to break symmetry
            if restart > 0:
                pts += rng.normal(0, spacing * 0.01, size=pts.shape)
            
            score = _compute_score(pts)
            candidates.append((score, pts))
    
    if not candidates:
        # Fallback: simple grid
        pts = np.zeros((n_points, 2))
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            r = 0.5 * np.sqrt(i + 1) / np.sqrt(n_points)
            pts[i] = [r * np.cos(angle), r * np.sin(angle)]
        return pts
    
    # Return the best candidate
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1].copy()


def _local_optimize(pts, max_iter=1000, tol=1e-12):
    """Optimize positions to maximize the minimum distance."""
    n = len(pts)
    pts = pts.copy()
    
    # Use a smooth approximation of the min function
    # We want to maximize min(min_pairwise_dist/2, min_boundary_dist)
    # This is equivalent to minimizing -min(...)
    
    def objective(x):
        p = x.reshape(n, 2)
        # Boundary constraint: p should be inside unit circle
        boundary_dist = 1.0 - np.linalg.norm(p, axis=1)
        min_boundary = np.min(boundary_dist)
        
        if n > 1:
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists)
        else:
            min_pair = np.inf
        
        score = min(min_boundary, min_pair / 2.0)
        
        # Penalize invalid configurations
        if min_boundary < 0 or min_pair < 0:
            return -score - 10.0 * (max(0, -min_boundary) + max(0, -min_pair))
        
        return -score
    
    # Use L-BFGS-B with bounds to keep points in reasonable range
    bounds = [(-1.0, 1.0)] * (2 * n)
    
    result = minimize(
        objective,
        pts.flatten(),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol}
    )
    
    return result.x.reshape(n, 2)


def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Generate initial hexagonal packing
    pts = _hex_init(N, n_restarts=3, seed=42)
    
    # Local optimization
    best_pts = pts.copy()
    best_score = _compute_score(best_pts)
    
    # Try optimization with different initializations
    for restart in range(3):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            current_pts = best_pts.copy()
        else:
            current_pts = _hex_init(N, n_restarts=1, seed=42 + restart * 100)
        
        # Local optimization
        optimized = _local_optimize(current_pts, max_iter=500, tol=1e-10)
        score = _compute_score(optimized)
        
        if score > best_score:
            best_score = score
            best_pts = optimized
    
    # Final polish if time allows
    if time.time() < deadline - 10:
        final = _local_optimize(best_pts, max_iter=1000, tol=1e-12)
        final_score = _compute_score(final)
        if final_score > best_score:
            best_pts = final
    
    return best_pts
# EVOLVE-BLOCK-END
