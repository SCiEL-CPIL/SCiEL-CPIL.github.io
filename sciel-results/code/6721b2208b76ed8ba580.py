"""Optimized program for covering the sphere with 130 equal caps."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import time


def _spherical_grid(n_points=20000):
    """Generate a dense spherical grid using Fibonacci spiral."""
    i = np.arange(n_points) + 0.5
    z = 1.0 - 2.0 * i / n_points
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _covering_radius_exact(points, grid=None):
    """Compute the exact covering radius in radians using a dense grid."""
    if grid is None:
        grid = _spherical_grid()
    
    # Find nearest neighbor for each grid point
    tree = cKDTree(points)
    dists, _ = tree.query(grid)
    
    # Convert Euclidean distances to angular distances
    max_d = np.max(dists)
    if max_d >= 2.0:
        return np.pi
    return 2.0 * np.arcsin(np.clip(max_d / 2.0, -1.0, 1.0))


def _smooth_objective(x, grid, temperature=0.1):
    """
    Smooth, differentiable objective function using soft-min of squared distances.
    This creates a smooth landscape for gradient-based optimization.
    """
    pts = x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    pts = pts / norms
    
    # Compute squared Euclidean distances from each grid point to each sphere point
    # grid shape: (M, 3), pts shape: (N, 3)
    # dist_sq[i,j] = ||grid[i] - pts[j]||^2 = 2 - 2 * dot(grid[i], pts[j])
    dots = grid @ pts.T  # (M, N)
    dist_sq = 2.0 - 2.0 * dots  # (M, N)
    
    # Use soft-min (log-sum-exp trick) to get a smooth approximation of max distance
    # For each grid point, find the nearest point (min dist_sq)
    # soft_min_j(dist_sq[i,j]) = -log(sum_j exp(-dist_sq[i,j] / temperature)) * temperature
    # This is smooth and differentiable
    
    # Clip to avoid numerical issues
    dist_sq = np.maximum(dist_sq, 0.0)
    
    # Soft min over points for each grid point
    scaled = -dist_sq / temperature
    # Use log-sum-exp for numerical stability
    max_scaled = np.max(scaled, axis=1, keepdims=True)
    log_sum_exp = max_scaled.squeeze() + np.log(np.sum(np.exp(scaled - max_scaled), axis=1))
    soft_min_dist_sq = -temperature * log_sum_exp
    
    # The covering radius is the max over grid points of the nearest distance
    # We want to minimize the max, so we can use a smooth max approximation
    # max_i(soft_min_dist_sq[i]) ≈ -log(sum_i exp(-soft_min_dist_sq[i] / temp2)) * temp2
    temp2 = temperature * 0.5
    scaled2 = -soft_min_dist_sq / temp2
    max_scaled2 = np.max(scaled2)
    log_sum_exp2 = max_scaled2 + np.log(np.sum(np.exp(scaled2 - max_scaled2)))
    smooth_max_dist_sq = -temp2 * log_sum_exp2
    
    # Convert squared distance to angular distance
    # d = sqrt(dist_sq), angle = 2 * arcsin(d/2)
    d = np.sqrt(np.maximum(smooth_max_dist_sq, 0.0))
    if d >= 2.0:
        return np.pi
    return 2.0 * np.arcsin(np.clip(d / 2.0, -1.0, 1.0))


def _initial_points_fibonacci():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _initial_points_random(seed=42):
    """Generate random initial points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _perturb_points(points, sigma=0.1, seed=42):
    """Perturb points by adding Gaussian noise and re-normalizing."""
    rng = np.random.RandomState(seed)
    noise = sigma * rng.randn(*points.shape)
    perturbed = points + noise
    norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return perturbed / norms


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    timeout = 115.0  # Leave 5s buffer
    
    # Precompute grid for objective function
    grid = _spherical_grid(n_points=20000)
    
    # Try multiple initial configurations
    initial_configs = []
    
    # Fibonacci spiral
    initial_configs.append(_initial_points_fibonacci())
    
    # Random configurations with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        if time.time() - start_time > timeout - 60:
            break
        initial_configs.append(_initial_points_random(seed=seed))
    
    best_points = initial_configs[0].copy()
    best_radius = _covering_radius_exact(best_points, grid)
    
    try:
        # Phase 1: For each initial config, run L-BFGS-B with annealed temperature
        for init_idx, init_pts in enumerate(initial_configs):
            if time.time() - start_time > timeout:
                break
            
            current_points = init_pts.copy()
            current_radius = _covering_radius_exact(current_points, grid)
            
            # If this initial config is already better, update
            if current_radius < best_radius:
                best_points = current_points.copy()
                best_radius = current_radius
            
            # Annealed temperature optimization
            temperatures = [0.3, 0.1, 0.05, 0.02, 0.01]
            
            for temp in temperatures:
                if time.time() - start_time > timeout:
                    break
                
                def obj_smooth(x):
                    return _smooth_objective(x, grid, temperature=temp)
                
                x0 = current_points.flatten()
                
                result = minimize(
                    obj_smooth,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                if time.time() - start_time > timeout:
                    break
                
                opt_points = result.x.reshape(N, 3)
                norms = np.linalg.norm(opt_points, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)
                opt_points = opt_points / norms
                
                opt_radius = _covering_radius_exact(opt_points, grid)
                
                if opt_radius < current_radius:
                    current_points = opt_points
                    current_radius = opt_radius
                
                if current_radius < best_radius:
                    best_points = current_points.copy()
                    best_radius = current_radius
        
        # Phase 2: Local search with small perturbations from best solution
        if time.time() - start_time < timeout - 10:
            for trial in range(5):
                if time.time() - start_time > timeout:
                    break
                
                sigma = 0.05 * (0.5 ** trial)
                perturbed = _perturb_points(best_points, sigma=sigma, seed=trial + 100)
                
                def obj_smooth_local(x):
                    return _smooth_objective(x, grid, temperature=0.01)
                
                x0 = perturbed.flatten()
                
                result = minimize(
                    obj_smooth_local,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 60, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                if time.time() - start_time > timeout:
                    break
                
                opt_points = result.x.reshape(N, 3)
                norms = np.linalg.norm(opt_points, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)
                opt_points = opt_points / norms
                
                opt_radius = _covering_radius_exact(opt_points, grid)
                
                if opt_radius < best_radius:
                    best_points = opt_points
                    best_radius = opt_radius
        
        # Phase 3: Final Nelder-Mead refinement
        if time.time() - start_time < timeout - 5:
            def obj_exact(x):
                pts = x.reshape(N, 3)
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)
                pts = pts / norms
                return _covering_radius_exact(pts, grid)
            
            x0 = best_points.flatten()
            result_nm = minimize(
                obj_exact,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if time.time() - start_time < timeout:
                opt_points_nm = result_nm.x.reshape(N, 3)
                norms = np.linalg.norm(opt_points_nm, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)
                opt_points_nm = opt_points_nm / norms
                
                opt_radius_nm = _covering_radius_exact(opt_points_nm, grid)
                if opt_radius_nm < best_radius:
                    best_points = opt_points_nm
                    best_radius = opt_radius_nm
                    
    except Exception:
        pass
    
    # Ensure points are on the unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    best_points = best_points / norms
    
    # Check for coincident points and hemisphere issues
    tree = cKDTree(best_points)
    dists, _ = tree.query(best_points, k=2)
    if np.min(dists[:, 1]) < 1e-6:
        best_points = _initial_points_fibonacci()
    
    # Final check: ensure points are well-distributed (not in a hemisphere)
    # Check that the convex hull contains the origin
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(best_points)
        # Check if origin is inside the hull by checking barycentric coordinates
        # Simple check: verify that points span all octants
        for axis in range(3):
            if np.all(best_points[:, axis] > 0) or np.all(best_points[:, axis] < 0):
                raise ValueError("Points are in a hemisphere")
    except Exception:
        # If convex hull check fails, fall back to Fibonacci
        best_points = _initial_points_fibonacci()
    
    return best_points
# EVOLVE-BLOCK-END
