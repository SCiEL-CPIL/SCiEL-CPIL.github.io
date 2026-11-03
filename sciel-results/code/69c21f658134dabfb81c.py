"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 300
DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist, squareform


def _hex_init(n_points):
    """Generate a hexagonal lattice initialization inside the unit disk."""
    # Estimate spacing to get roughly n_points in unit disk
    # Area of hex lattice point ~ sqrt(3)/2 * s^2
    # We want ~n_points in area pi
    s = np.sqrt(np.pi / (n_points * np.sqrt(3.0) / 2.0))
    
    best_points = None
    best_count = 0
    
    for trial in range(20):
        spacing = s * (1.0 - 0.02 * trial)
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Keep points that fit with margin
        margin = spacing / 2.0
        mask = radius <= 1.0 - margin
        points = points[mask]
        
        if len(points) >= n_points:
            # Sort by radius, take closest to center
            idx = np.argsort(radius[mask], kind='stable')[:n_points]
            pts = points[idx]
            if best_points is None or len(pts) > best_count:
                best_points = pts
                best_count = len(pts)
    
    if best_points is not None:
        return best_points
    
    # Fallback: random
    return np.random.randn(n_points, 2) * 0.9


def _compute_radius(points):
    """Compute the maximum common radius for given points."""
    if len(points) < 2:
        return 0.0
    
    # Min distance between any two centers / 2
    d = pdist(points)
    min_pair_dist = d.min() if len(d) > 0 else np.inf
    
    # Min distance to boundary
    norms = np.linalg.norm(points, axis=1)
    min_boundary_dist = (1.0 - norms).min()
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return max(r, 0.0)


def _soft_min_objective(params, n_points, temperature):
    """
    Objective function for optimization.
    params: flattened array of shape (2*n_points,)
    Returns residuals that penalize:
    1. Points outside the unit disk
    2. Points too close to each other (soft min of distances)
    """
    points = params.reshape(n_points, 2)
    
    # Boundary constraint: points should be within unit disk
    norms = np.linalg.norm(points, axis=1)
    boundary_violation = np.maximum(0.0, norms - 1.0)
    
    # Pairwise distances
    d = pdist(points)
    
    # Soft-min of distances: we want all distances to be large
    # Use log-sum-exp trick for numerical stability
    # Penalize when distances are small
    # target_min_dist is set based on current best estimate
    # We'll use a fixed target that gets updated
    
    # For the objective, we want to maximize the minimum distance
    # Use: -log(sum(exp(-d_i / T))) * T as a soft min
    # This is approximately the min when T is small
    
    if len(d) == 0:
        return np.concatenate([boundary_violation, np.zeros(1)])
    
    # Soft min of pairwise distances
    # exp(-d/T) is large when d is small
    # log(sum(exp(-d/T))) * T ≈ min(d) when T is small
    exp_neg_d_over_T = np.exp(-d / temperature)
    log_sum_exp = np.log(np.sum(exp_neg_d_over_T))
    soft_min_dist = -temperature * log_sum_exp
    
    # We want to maximize soft_min_dist, so minimize -soft_min_dist
    # Also penalize boundary violations
    # Scale boundary violation to be comparable
    boundary_penalty = 10.0 * boundary_violation
    
    residuals = np.concatenate([
        boundary_penalty,
        np.array([-soft_min_dist])
    ])
    
    return residuals


def _optimize_from_init(points_init, time_limit):
    """
    Optimize starting from points_init.
    Returns optimized points.
    """
    n = len(points_init)
    params = points_init.flatten()
    
    # Adaptive temperature schedule
    T0 = 0.05  # initial temperature
    T_min = 0.001
    
    # First, get a baseline radius
    base_r = _compute_radius(points_init)
    
    # Try a few different temperatures
    best_points = points_init.copy()
    best_r = base_r
    
    for T in [T0, T0 * 0.5, T0 * 0.25, T0 * 0.1]:
        if time.time() > time_limit:
            break
        
        def objective(p):
            return _soft_min_objective(p, n, T)
        
        try:
            result = least_squares(
                objective,
                params,
                method='trf',
                max_nfev=5000,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10
            )
            
            opt_points = result.x.reshape(n, 2)
            
            # Validate: check if all points are inside and finite
            if not np.all(np.isfinite(opt_points)):
                continue
            if np.any(np.linalg.norm(opt_points, axis=1) > 1.0 + 1e-6):
                # Clip points that are slightly outside
                norms = np.linalg.norm(opt_points, axis=1)
                mask = norms > 1.0
                if np.any(mask):
                    opt_points[mask] *= (1.0 - 1e-9) / norms[mask]
            
            r = _compute_radius(opt_points)
            if r > best_r:
                best_r = r
                best_points = opt_points.copy()
                
        except Exception:
            continue
    
    return best_points, best_r


def construct():
    """
    Construct 300 circle centers inside the unit disk to maximize the common radius.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Initial hexagonal lattice
    points_init = _hex_init(N)
    
    # Optimize from initial configuration
    points, r = _optimize_from_init(points_init, deadline - 5.0)
    
    elapsed = time.time() - start_time
    
    # If we have time, try a second round with perturbation
    if elapsed < DEADLINE - 10.0:
        # Perturb the best solution and re-optimize
        np.random.seed(42)
        perturbation_scale = 0.01
        points_perturbed = points + np.random.randn(N, 2) * perturbation_scale
        
        # Ensure perturbed points are inside the disk
        norms = np.linalg.norm(points_perturbed, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            points_perturbed[mask] *= (1.0 - 1e-9) / norms[mask]
        
        points2, r2 = _optimize_from_init(points_perturbed, deadline - 5.0)
        
        if r2 > r:
            points = points2
            r = r2
    
    # Final validation and cleanup
    # Ensure all points are strictly inside the unit disk
    norms = np.linalg.norm(points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        points[mask] *= (1.0 - 1e-9) / norms[mask]
    
    # Ensure finite values
    if not np.all(np.isfinite(points)):
        # Fallback to hex init
        points = _hex_init(N)
    
    return points
# EVOLVE-BLOCK-END
