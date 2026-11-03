"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_radius(pts):
    """Compute the maximum common radius for given centers."""
    # Boundary constraint: min distance from any center to the square edge
    # Square is [-0.5, 0.5]^2, so distance to edge is min(0.5 - |x|, 0.5 - |y|)
    boundary = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
    r_boundary = np.min(boundary)
    
    # Pairwise distances
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pairs = np.min(dists) / 2.0
    
    return min(r_boundary, r_pairs)


def _negative_radius(pts_flat):
    """Objective function: negative of the radius (to minimize)."""
    pts = pts_flat.reshape(-1, 2)
    return -_compute_radius(pts)


def _soft_negative_radius(pts_flat, beta=100.0):
    """Smooth surrogate using log-sum-exp for the min operation."""
    pts = pts_flat.reshape(-1, 2)
    n = len(pts)
    
    # Boundary terms
    boundary = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
    
    # Pairwise distances
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Collect all constraint values (we want the minimum)
    # For soft min: -log(sum(exp(-x/beta))) * beta approximates min(x)
    # We have boundary constraints and pairwise/2 constraints
    
    # Boundary soft-min
    b_neg = -boundary / beta
    b_neg_max = np.max(b_neg)
    soft_min_boundary = -beta * (b_neg_max + np.log(np.sum(np.exp(b_neg - b_neg_max))))
    
    # Pairwise soft-min (only upper triangle to avoid duplication)
    triu_idx = np.triu_indices(n, k=1)
    pair_dists = np.sqrt(dists_sq[triu_idx]) / 2.0
    p_neg = -pair_dists / beta
    p_neg_max = np.max(p_neg)
    soft_min_pairs = -beta * (p_neg_max + np.log(np.sum(np.exp(p_neg - p_neg_max))))
    
    # Combine: we want min(soft_min_boundary, soft_min_pairs)
    # Use another soft-min
    combined = np.array([soft_min_boundary, soft_min_pairs])
    c_neg = -combined / beta
    c_neg_max = np.max(c_neg)
    result = -beta * (c_neg_max + np.log(np.sum(np.exp(c_neg - c_neg_max))))
    
    return -result


def _init_7x7_grid():
    """Initialize with 7x7 grid pattern (49 points) plus one extra point."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    
    # We have 49 points, need 50. Add one more point.
    # The 7x7 grid has spacing 1/7 ≈ 0.142857, so radius = 1/14 ≈ 0.071429
    # This is already slightly better than the known best of 0.071377103865
    
    # For the 50th point, we can try to find a position that doesn't reduce the radius
    # One approach: slightly perturb one of the existing points and add a new one
    # Or: use a different structure
    
    # Let's try: take the 7x7 grid and add a point in one of the "gaps"
    # Actually, with 49 points in 7x7, the optimal 50th point placement is tricky.
    # Let's start with the 7x7 grid and let optimization handle the 50th point.
    
    # Start with 7x7 grid (49 points) and add a 50th point at origin
    # (will be moved by optimization)
    pts = grid.copy()
    pts = np.vstack([pts, np.array([0.0, 0.0])])
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initialize with 7x7 grid + one extra point
    pts = _init_7x7_grid()
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    # Try multiple random perturbations and optimize
    rng = np.random.RandomState(42)
    
    n_optimizations = 0
    max_optimizations = 50
    
    while time.time() < deadline and n_optimizations < max_optimizations:
        n_optimizations += 1
        
        # Create a perturbed version
        if n_optimizations == 1:
            # First: use the initial grid
            x0 = pts.copy()
        else:
            # Subsequent: perturb the best found so far
            noise_scale = 0.01 / np.sqrt(n_optimizations)
            x0 = best_pts + rng.randn(50, 2) * noise_scale
        
        # Optimize using Nelder-Mead
        try:
            result = minimize(
                _negative_radius,
                x0.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            new_pts = result.x.reshape(-1, 2)
            new_r = _compute_radius(new_pts)
            
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
                
        except Exception:
            pass
        
        # Check deadline
        if time.time() > deadline - 10:
            break
    
    # Final polish: run a few more Nelder-Mead iterations with tighter tolerance
    if time.time() < deadline - 5:
        try:
            result = minimize(
                _negative_radius,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            
            new_pts = result.x.reshape(-1, 2)
            new_r = _compute_radius(new_pts)
            
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
        except Exception:
            pass
    
    # Ensure all points are within the square (clip with small margin)
    best_pts = np.clip(best_pts, -0.5 + 1e-15, 0.5 - 1e-15)
    
    # Final validity check
    final_r = _compute_radius(best_pts)
    if final_r <= 0 or not np.all(np.isfinite(best_pts)):
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_pts
# EVOLVE-BLOCK-END
