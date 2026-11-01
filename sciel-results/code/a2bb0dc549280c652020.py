import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import warnings

N = 130
_time_start = None
_time_limit = 148.0  # 150s wall-clock, leave 2s margin

def _covering_radius_from_hull(points):
    """
    Compute the exact covering radius from the convex hull.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    max_radius = 0.0
    for simplex in hull.simplices:
        verts = points[simplex]
        a, b, c = verts[0], verts[1], verts[2]
        normal = np.cross(b - a, c - a)
        norm_val = np.linalg.norm(normal)
        
        if norm_val < 1e-12:
            continue
            
        normal = normal / norm_val
        
        centroid = np.mean(verts, axis=0)
        if np.dot(normal, centroid) < 0:
            normal = -normal
        
        dots = np.clip(np.dot(verts, normal), -1.0, 1.0)
        angles = np.arccos(dots)
        
        facet_radius = np.max(angles)
        max_radius = max(max_radius, facet_radius)
    
    return max_radius

def _soft_covering_radius_and_grad(points, grid_points, sigma=0.05):
    """
    Vectorized soft covering radius and its gradient.
    """
    dots = grid_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    
    neg_scaled = -angles / sigma
    max_neg = np.max(neg_scaled, axis=1, keepdims=True)
    exp_neg = np.exp(neg_scaled - max_neg)
    weights = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)
    
    soft_min_vals = np.sum(weights * angles, axis=1)
    
    scaled_max = soft_min_vals / sigma
    max_scaled = np.max(scaled_max)
    exp_max = np.exp(scaled_max - max_scaled)
    max_weights = exp_max / np.sum(exp_max)
    soft_max_val = sigma * (max_scaled + np.log(np.sum(exp_max)))
    
    denom = np.sqrt(np.maximum(1.0 - dots * dots, 1e-18))
    
    coeff = max_weights[:, np.newaxis] * weights
    alpha = coeff / denom
    
    term1 = -(alpha.T @ grid_points)
    
    beta = alpha * dots
    beta_sum = np.sum(beta, axis=0)
    term2 = beta_sum[:, np.newaxis] * points
    
    grad = term1 + term2
    
    return soft_max_val, grad

def _lloyd_step(points, grid_points, n_iters=3):
    """
    Perform Lloyd's algorithm iterations using grid-based Voronoi.
    """
    for _ in range(n_iters):
        dots = grid_points @ points.T
        angles = np.arccos(np.clip(dots, -1.0, 1.0))
        nearest = np.argmin(angles, axis=1)
        
        counts = np.bincount(nearest, minlength=N).astype(np.float64)
        sums = np.zeros((N, 3))
        for d in range(3):
            sums[:, d] = np.bincount(nearest, weights=grid_points[:, d], minlength=N)
        
        safe_counts = np.where(counts > 0, counts, 1.0)
        new_points = sums / safe_counts[:, np.newaxis]
        
        empty = counts == 0
        if np.any(empty):
            new_points[empty] = points[empty]
        
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        new_points = new_points / norms
        
        diff = np.linalg.norm(new_points - points, axis=1)
        points = new_points
        
        if np.max(diff) < 1e-7:
            break
    
    return points

def _local_optimize(points, grid_points, max_iters=50):
    """
    Fast local optimization using L-BFGS-B on the soft grid radius.
    """
    def objective_and_grad(x_flat):
        pts = x_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
        val, grad = _soft_covering_radius_and_grad(pts, grid_points, sigma=0.02)
        return val, grad.flatten()
    
    x0 = points.flatten()
    try:
        res = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        if res.x is not None:
            pts = res.x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            pts = pts / norms
            return pts
    except Exception:
        pass
    return points

def construct():
    global _time_start
    _time_start = time.time()
    
    # Generate grid on sphere
    n_grid = 20000 # Smaller grid for speed in local opt
    i = np.arange(n_grid) + 0.5
    z_grid = 1.0 - 2.0 * i / n_grid
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    grid_points = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    rng = np.random.default_rng(42)
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    spiral_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Initial local optimization
    points = _local_optimize(spiral_points, grid_points, max_iters=100)
    
    # Evaluate exact radius
    best_score = _covering_radius_from_hull(points)
    best_points = points.copy()
    
    # Basin Hopping Loop
    # Perturbation strength starts moderate and can be adjusted
    perturb_scale = 0.1
    
    while time.time() - _time_start < _time_limit - 5.0:
        # Create a perturbed copy
        perturbed = best_points + rng.normal(0, perturb_scale, size=(N, 3))
        # Normalize
        perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
        
        # Local optimization on the perturbed points
        candidate = _local_optimize(perturbed, grid_points, max_iters=50)
        
        # Evaluate exact radius of candidate
        candidate_score = _covering_radius_from_hull(candidate)
        
        if candidate_score < best_score:
            best_score = candidate_score
            best_points = candidate.copy()
            # Reduce perturbation scale slightly as we improve? Or keep it.
            # perturb_scale *= 0.95
            if best_score < 0.197: # ~11.3 degrees in radians is 0.197? No.
                # 11.316 degrees = 0.1975 radians
                # If we beat the known best, we can stop or keep going.
                pass
        else:
            # If we didn't improve, maybe increase perturbation to escape local min
            # perturb_scale *= 1.1
            pass
            
        # Optional: Try a different initial random start every few iterations?
        # No, Basin Hopping is better.

    return best_points
