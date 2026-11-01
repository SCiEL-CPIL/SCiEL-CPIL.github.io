import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import warnings

N = 130

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
    Returns (value, grad) where grad has shape (N, 3).
    """
    # Compute angular distances
    dots = grid_points @ points.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (M, N)
    
    # For each grid point, find minimum angle
    min_angles = np.min(angles, axis=1)  # (M,)
    nearest_idx = np.argmin(angles, axis=1)  # (M,)
    
    # Soft max via log-sum-exp
    scaled = min_angles / sigma
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / np.sum(exp_scaled)  # (M,)
    soft_max_val = sigma * (max_scaled + np.log(np.sum(exp_scaled)))
    
    # Compute gradient: for each grid point i, contributes to point nearest_idx[i]
    # d(angle_i)/d(point_j) where j = nearest_idx[i]
    # angle_i = arccos(dot(g_i, p_j))
    # d/dp_j arccos(dot(g, p)) = -g / sqrt(1 - dot(g,p)^2) projected onto tangent plane
    
    # Get the relevant grid points and configuration points
    g_sel = grid_points[np.arange(len(weights)), :]  # (M, 3) - all grid points
    p_sel = points[nearest_idx]  # (M, 3) - the nearest configuration point for each grid point
    
    dot_sel = dots[np.arange(len(weights)), nearest_idx]  # (M,)
    dot_sel = np.clip(dot_sel, -1.0 + 1e-9, 1.0 - 1e-9)
    
    # Raw gradient: -g / sqrt(1 - dot^2)
    denom = np.sqrt(1.0 - dot_sel * dot_sel)
    grad_raw = -g_sel / denom[:, np.newaxis]  # (M, 3)
    
    # Project onto tangent plane: grad_tan = grad_raw - (grad_raw . p_sel) * p_sel
    dot_grad_p = np.sum(grad_raw * p_sel, axis=1, keepdims=True)  # (M, 1)
    grad_tan = grad_raw - dot_grad_p * p_sel  # (M, 3)
    
    # Weighted sum
    weighted_grad = grad_tan * weights[:, np.newaxis]  # (M, 3)
    
    # Scatter-add to get gradient for each configuration point
    grad = np.zeros((N, 3))
    np.add.at(grad, nearest_idx, weighted_grad)
    
    return soft_max_val, grad

def _lloyd_step(points, grid_points, n_iters=3):
    """
    Perform Lloyd's algorithm iterations using grid-based Voronoi.
    """
    for _ in range(n_iters):
        dots = grid_points @ points.T
        nearest = np.argmin(np.arccos(np.clip(dots, -1.0, 1.0)), axis=1)
        
        new_points = np.zeros_like(points)
        
        for j in range(N):
            mask = (nearest == j)
            if np.sum(mask) > 0:
                new_points[j] = np.mean(grid_points[mask], axis=0)
            else:
                new_points[j] = points[j]
        
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        new_points = new_points / norms
        
        diff = np.linalg.norm(new_points - points, axis=1)
        points = new_points
        
        if np.max(diff) < 1e-6:
            break
    
    return points

def construct():
    start_time = time.time()
    total_time_limit = 145.0  # Leave 5s margin for 150s limit
    
    # Generate grid on sphere
    n_grid = 15000
    i = np.arange(n_grid) + 0.5
    z_grid = 1.0 - 2.0 * i / n_grid
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    grid_points = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    best_score = np.inf
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Generate starting configurations
    starts = []
    
    # Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    spiral_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    starts.append(spiral_points)
    
    # Random points
    for _ in range(5):
        rp = rng.normal(size=(N, 3))
        rp /= np.linalg.norm(rp, axis=1, keepdims=True)
        starts.append(rp)
    
    # Process each start
    for idx, start_pts in enumerate(starts):
        if time.time() - start_time > total_time_limit - 25.0:
            break
        
        points = start_pts.copy()
        
        # Step 1: Brief Lloyd's iterations
        remaining_time = total_time_limit - (time.time() - start_time)
        lloyd_iters = min(8, max(3, int(remaining_time / 10)))
        
        for _ in range(lloyd_iters):
            if time.time() - start_time > total_time_limit - 20.0:
                break
            points = _lloyd_step(points, grid_points, n_iters=2)
        
        # Step 2: Multi-stage gradient optimization with sigma annealing
        if time.time() - start_time < total_time_limit - 15.0:
            sigmas = [0.10, 0.05, 0.02]
            
            for sigma in sigmas:
                if time.time() - start_time > total_time_limit - 12.0:
                    break
                
                def objective_and_grad(x_flat, sigma=sigma):
                    pts = x_flat.reshape(N, 3)
                    norms = np.linalg.norm(pts, axis=1, keepdims=True)
                    norms = np.where(norms < 1e-12, 1.0, norms)
                    pts = pts / norms
                    
                    val, grad = _soft_covering_radius_and_grad(pts, grid_points, sigma=sigma)
                    return val, grad.flatten()
                
                x0 = points.flatten()
                
                try:
                    res = minimize(
                        objective_and_grad,
                        x0,
                        jac=True,
                        method='L-BFGS-B',
                        options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False}
                    )
                    if res.x is not None:
                        points = res.x.reshape(N, 3)
                        norms = np.linalg.norm(points, axis=1, keepdims=True)
                        norms = np.where(norms < 1e-12, 1.0, norms)
                        points = points / norms
                except Exception:
                    pass
        
        # Step 3: Final polish with very small sigma
        if time.time() - start_time < total_time_limit - 5.0:
            def objective_and_grad_polish(x_flat):
                pts = x_flat.reshape(N, 3)
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                norms = np.where(norms < 1e-12, 1.0, norms)
                pts = pts / norms
                
                val, grad = _soft_covering_radius_and_grad(pts, grid_points, sigma=0.005)
                return val, grad.flatten()
            
            try:
                res = minimize(
                    objective_and_grad_polish,
                    points.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
                if res.x is not None:
                    points = res.x.reshape(N, 3)
                    norms = np.linalg.norm(points, axis=1, keepdims=True)
                    norms = np.where(norms < 1e-12, 1.0, norms)
                    points = points / norms
            except Exception:
                pass
        
        # Evaluate exact covering radius
        score = _covering_radius_from_hull(points)
        
        if score < best_score:
            best_score = score
            best_points = points.copy()
        
        if best_score < 0.1975:
            break
    
    if best_points is None:
        best_points = spiral_points
        best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
