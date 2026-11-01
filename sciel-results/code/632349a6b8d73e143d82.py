import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _get_grid(n_grid=20000):
    """Generate a dense spherical grid for the surrogate objective."""
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _hull_radius(pts):
    """Compute the exact covering radius using the convex hull method."""
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.pi

        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        return np.max(radii)
    except Exception:
        return np.pi

def _soft_hull_objective_and_grad(pts, grid, temperature):
    """
    Compute the soft-hull objective: log-sum-exp of squared distances to grid points.
    This approximates the maximum distance (covering radius) in a differentiable way.
    
    Args:
        pts: (N, 3) array of points on the sphere
        grid: (M, 3) array of grid points
        temperature: parameter for the log-sum-exp (lower = sharper max)
    
    Returns:
        obj: scalar objective value
        grad: (N, 3) gradient
    """
    # Ensure points are on the sphere
    pts = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + _EPS)
    
    # Compute squared distances: dist_sq[i, j] = ||grid[i] - pts[j]||^2
    # Using dot products for efficiency: ||a-b||^2 = 2 - 2 a.b for unit vectors
    dots = grid @ pts.T  # (M, N)
    dist_sq = 2.0 - 2.0 * dots  # (M, N)
    
    # For each grid point i, find the minimum squared distance to any point j
    # min_dist_sq[i] = min_j dist_sq[i, j]
    # We use log-sum-exp to approximate the max of min_dist_sq
    # Actually, we want to minimize the max over grid points of the min distance.
    # Let f_i = min_j dist_sq[i, j]. We want min max_i f_i.
    # Approximate max_i f_i with log-sum-exp(f_i / T) * T.
    
    # First, get min distance for each grid point
    min_dist_sq = np.min(dist_sq, axis=1)  # (M,)
    
    # Log-sum-exp trick for numerical stability
    max_val = np.max(min_dist_sq)
    if max_val > 1e10:
        return max_val, np.zeros_like(pts)
    
    # scaled = min_dist_sq / temperature
    scaled = min_dist_sq / temperature
    max_scaled = np.max(scaled)
    
    # log_sum_exp = max_scaled + log(sum(exp(scaled - max_scaled)))
    exp_vals = np.exp(scaled - max_scaled)
    log_sum = np.sum(exp_vals)
    log_sum_exp_val = max_scaled + np.log(log_sum + _EPS)
    
    obj = temperature * log_sum_exp_val
    
    # Gradient calculation
    # d(obj)/d(pts[k]) = sum_{i} d(f_i)/d(pts[k]) * weight_i
    # where f_i = min_j dist_sq[i, j]
    # and weight_i = exp(scaled_i - max_scaled) / sum(exp(scaled_j - max_scaled))
    
    # weights for each grid point
    weights = exp_vals / (log_sum + _EPS)  # (M,)
    
    # For each grid point i, the nearest point index is argmin(dist_sq[i, :])
    min_idx = np.argmin(dist_sq, axis=1)  # (M,)
    
    # d(dist_sq[i, j])/d(pts[j]) = -2 * grid[i]
    # So d(f_i)/d(pts[k]) = -2 * grid[i] if min_idx[i] == k, else 0
    
    grad = np.zeros_like(pts)
    # Accumulate gradients
    # For each i, add -2 * grid[i] * weights[i] to grad[min_idx[i]]
    # Vectorized: use np.add.at
    weighted_grid = -2.0 * grid * weights[:, np.newaxis]  # (M, 3)
    np.add.at(grad, min_idx, weighted_grid)
    
    # Project gradient to tangent space? 
    # Since we are optimizing on the sphere, the gradient in Euclidean space 
    # should be projected. However, L-BFGS-B operates in Euclidean space.
    # If we use L-BFGS-B, we should just return the Euclidean gradient.
    # The optimizer will step in Euclidean space, and we re-normalize in the next iteration.
    
    return obj, grad.flatten()

def _riemannian_grad_descent(pts, grid, deadline, initial_step=0.01, temperature=0.1):
    """
    Perform Riemannian Gradient Descent on the sphere.
    """
    best_pts = pts.copy()
    best_obj = _hull_radius(pts)  # Use true metric for tracking best
    step = initial_step
    no_improve = 0
    rng = np.random.default_rng(123)
    
    # Decrease temperature over time to sharpen the max approximation
    temp_schedule = lambda t: max(0.01, temperature * np.exp(-t / 5.0))
    
    t_iter = 0
    while time.time() < deadline:
        t_iter += 1
        temp = temp_schedule(t_iter)
        
        obj, grad = _soft_hull_objective_and_grad(pts, grid, temp)
        grad = grad.reshape(N, 3)
        
        # Project gradient to tangent space of the sphere
        # grad_tangent = grad - (grad . pts) * pts
        dot_prod = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tan = grad - dot_prod * pts
        
        # Adaptive step size
        # Normalize gradient
        grad_norm = np.linalg.norm(grad_tan, axis=1, keepdims=True)
        grad_tan_norm = grad_tan / (grad_norm + _EPS)
        
        # Update
        new_pts = pts - step * grad_tan_norm
        new_pts /= (np.linalg.norm(new_pts, axis=1, keepdims=True) + _EPS)
        
        # Check improvement using true metric occasionally to save time
        if t_iter % 5 == 0:
            new_r = _hull_radius(new_pts)
            if new_r < best_obj - 1e-10:
                best_obj = new_r
                best_pts = new_pts.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 10:
                    step *= 0.9
                    no_improve = 0
                    if step < 1e-5:
                        break
        
        pts = new_pts
        
    return best_pts, best_obj

def _polish_with_hull(pts, deadline):
    """
    Polish the points using subgradient descent on the actual hull metric.
    """
    best_pts = pts.copy()
    best_r = _hull_radius(pts)
    
    step = 0.02
    no_improve_count = 0
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Compute hull radii and poles
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            A = pts[faces[:, 0]]
            B = pts[faces[:, 1]]
            C = pts[faces[:, 2]]
            AB = B - A
            AC = C - A
            cross = np.cross(AB, AC)
            area = 0.5 * np.linalg.norm(cross, axis=1)
            valid = area > _EPS
            if not np.any(valid):
                break
            normals = cross[valid]
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            centroids = (A[valid] + B[valid] + C[valid]) / 3.0
            signs = np.sum(normals * centroids, axis=1, keepdims=True)
            normals = normals * np.sign(signs)
            normals = normals / (norms + _EPS)
            cos_theta = np.sum(normals * A[valid], axis=1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            radii = np.arccos(cos_theta)
        except Exception:
            break
        
        curr_r = np.max(radii)
        
        if curr_r < best_r - 1e-9:
            best_r = curr_r
            best_pts = pts.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 5:
                step *= 0.9
                no_improve_count = 0
                if step < 1e-6:
                    break
        
        worst_idx = np.argmax(radii)
        v_indices = faces[valid][worst_idx]
        pole = normals[worst_idx]
        
        cand = pts.copy()
        for v in v_indices:
            dir_vec = pole - pts[v]
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm > _EPS:
                dir_vec /= dir_norm
            else:
                dir_vec = rng.standard_normal(3)
                dir_vec /= (np.linalg.norm(dir_vec) + _EPS)
            
            cand[v] = pts[v] + step * dir_vec
            cand[v] /= (np.linalg.norm(cand[v]) + _EPS)
        
        cand_r = _hull_radius(cand)
        if cand_r < curr_r:
            pts = cand
        else:
            step *= 0.8
            
    return best_pts, best_r

def construct():
    deadline = time.time() + 140.0  # Safety margin
    grid = _get_grid(20000)
    
    best_pts = _initial_fib()
    best_r = _hull_radius(best_pts)
    
    # Strategy: Multiple restarts with Riemannian Gradient Descent
    n_restarts = 5
    time_per_restart = (deadline - 30.0) / n_restarts  # Leave 30s for polishing
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            pts = _initial_fib()
        else:
            # Perturb the best found so far
            rng = np.random.default_rng(42 + restart)
            pts = best_pts + rng.standard_normal((N, 3)) * 0.2
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Run Riemannian Gradient Descent
        new_pts, new_r = _riemannian_grad_descent(pts, grid, deadline - 30.0 - (n_restarts - restart - 1) * time_per_restart)
        
        if new_r < best_r:
            best_r = new_r
            best_pts = new_pts
    
    # Final polish with the true metric
    if time.time() < deadline - 5.0:
        best_pts, best_r = _polish_with_hull(best_pts, deadline)
    
    # Final normalization and validity check
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
