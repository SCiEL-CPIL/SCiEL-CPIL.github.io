"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist
import time


def _make_initial_points(n):
    """Create initial hexagonal lattice points inside the unit disk."""
    # Estimate spacing needed for n points in unit disk
    # Area of disk = pi, so avg area per point = pi/n
    # Hexagonal lattice: each point occupies ~ (sqrt(3)/2) * s^2 where s is spacing
    # So s ~ sqrt(2*pi/(n*sqrt(3)))
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    k = int(1.0 / s) + 2
    rows = [((i + 0.5 * (j % 2)) * s, j * s * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 0.95
    points = points[mask]
    if len(points) < n:
        # Fallback: denser grid
        xs = np.linspace(-0.9, 0.9, 30)
        ys = np.linspace(-0.9, 0.9, 30)
        grid = np.array([[x, y] for x in xs for y in ys])
        radius = np.hypot(grid[:, 0], grid[:, 1])
        grid = grid[radius <= 0.95]
        points = grid[:n]
    else:
        # Sort by distance from origin to get a nice distribution
        dist = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dist, kind='stable')
        points = points[idx[:n]]
    return points


def _compute_radius(centres):
    """Compute the packing radius for a given set of centres."""
    n = len(centres)
    if n < 2:
        return 1.0 - np.max(np.linalg.norm(centres, axis=1))
    dists = pdist(centres)
    min_pair_dist = np.min(dists)
    boundary_dists = 1.0 - np.linalg.norm(centres, axis=1)
    min_boundary_dist = np.min(boundary_dists)
    return min(min_pair_dist / 2.0, min_boundary_dist)


def _feasibility_residuals(centres_flat, r_target, n):
    """
    Compute residuals for feasibility check at a given r_target.
    Returns the squared violations (should be ~0 if feasible).
    """
    centres = centres_flat.reshape(n, 2)
    dists = pdist(centres)
    boundary_dists = 1.0 - np.linalg.norm(centres, axis=1)
    
    # Pair violations: want d_ij >= 2*r_target
    pair_violations = np.maximum(0, 2 * r_target - dists)
    # Boundary violations: want 1 - |c_i| >= r_target
    boundary_violations = np.maximum(0, r_target - boundary_dists)
    
    return np.concatenate([pair_violations, boundary_violations])


def _check_feasible(centres, r_target, n, tol=1e-6):
    """Check if configuration is feasible for radius r_target."""
    dists = pdist(centres)
    min_pair = np.min(dists)
    boundary = 1.0 - np.linalg.norm(centres, axis=1)
    min_boundary = np.min(boundary)
    return min(min_pair / 2.0, min_boundary) >= r_target - tol


def construct():
    """
    Construct centres of 300 equal circles inside the unit circle.
    Uses binary search on r with least_squares feasibility checks.
    """
    n = N
    time_start = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Initial points
    points = _make_initial_points(n)
    x0 = points.flatten()
    
    # First, get a baseline radius from the initial configuration
    r_baseline = _compute_radius(points)
    
    # Binary search on r
    r_low = 0.0
    r_high = min(0.5, 1.0 / np.sqrt(n) * 1.5)  # Upper bound estimate
    best_centres = points.copy()
    best_r = r_baseline
    
    # Do a few iterations of binary search
    max_iter = 30
    for it in range(max_iter):
        if time.time() - time_start > time_limit:
            break
        
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a feasible configuration for r_mid
        # Use least_squares to minimize violations
        def residuals(x):
            centres = x.reshape(n, 2)
            return _feasibility_residuals(x, r_mid, n)
        
        try:
            result = least_squares(
                residuals, x0, method='trf', max_nfev=5000,
                xtol=1e-10, ftol=1e-10, gtol=1e-10,
                bounds=([-0.999] * 2 * n, [0.999] * 2 * n)
            )
            new_centres = result.x.reshape(n, 2)
            
            # Check if this is feasible
            new_r = _compute_radius(new_centres)
            if new_r >= r_mid - 1e-8:
                # Feasible! Update best and raise lower bound
                if new_r > best_r:
                    best_r = new_r
                    best_centres = new_centres.copy()
                r_low = r_mid
                x0 = result.x  # Use this as starting point for next iteration
            else:
                # Not feasible, lower the upper bound
                r_high = r_mid
        except Exception:
            r_high = r_mid
    
    # Final polish: try to improve from best_centres
    if time.time() - time_start < time_limit - 10:
        # Do a local optimization to maximize the minimum distance
        # Use a different formulation: minimize the sum of squared shortfalls
        # from the current best radius
        
        def polish_residuals(x):
            centres = x.reshape(n, 2)
            dists = pdist(centres)
            boundary_dists = 1.0 - np.linalg.norm(centres, axis=1)
            
            # Estimate current radius
            r_est = min(np.min(dists) / 2.0, np.min(boundary_dists))
            
            # Penalize pairs that are too close relative to r_est
            pair_res = np.maximum(0, 2 * r_est - dists)
            # Penalize points too close to boundary
            bound_res = np.maximum(0, r_est - boundary_dists)
            
            return np.concatenate([pair_res, bound_res])
        
        try:
            result = least_squares(
                polish_residuals, best_centres.flatten(), method='trf',
                max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12,
                bounds=([-0.9999] * 2 * n, [0.9999] * 2 * n)
            )
            polished = result.x.reshape(n, 2)
            polished_r = _compute_radius(polished)
            if polished_r > best_r:
                best_r = polished_r
                best_centres = polished.copy()
        except Exception:
            pass
    
    # Final validation
    if not np.all(np.isfinite(best_centres)):
        # Fallback
        best_centres = _make_initial_points(n)
    
    # Ensure all points are strictly inside
    norms = np.linalg.norm(best_centres, axis=1)
    if np.any(norms >= 1.0):
        # Scale down slightly
        scale = 0.999 / np.max(norms)
        best_centres = best_centres * scale
    
    return best_centres
# EVOLVE-BLOCK-END
