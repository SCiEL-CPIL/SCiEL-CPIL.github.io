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
    Vectorized soft covering radius and its gradient using smooth soft-min over all points.
    For each grid point, we compute a weighted sum over all configuration points with
    weights from a soft-min of the angular distances. This gives a smooth gradient.
    Returns (value, grad) where grad has shape (N, 3).
    """
    # Compute angular distances: (M, N)
    dots = grid_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (M, N)
    
    # Soft-min: for each grid point, compute weighted average of angles
    # using softmax weights over the negative angles (to get min)
    # w_j = exp(-angle_j / sigma) / sum_k exp(-angle_k / sigma)
    neg_scaled = -angles / sigma  # (M, N)
    max_neg = np.max(neg_scaled, axis=1, keepdims=True)  # (M, 1)
    exp_neg = np.exp(neg_scaled - max_neg)  # (M, N)
    weights = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)  # (M, N)
    
    # Soft-min value: weighted sum of angles
    soft_min_vals = np.sum(weights * angles, axis=1)  # (M,)
    
    # Soft-max of soft-min values (to get covering radius approximation)
    scaled_max = soft_min_vals / sigma
    max_scaled = np.max(scaled_max)
    exp_max = np.exp(scaled_max - max_scaled)
    max_weights = exp_max / np.sum(exp_max)  # (M,)
    soft_max_val = sigma * (max_scaled + np.log(np.sum(exp_max)))
    
    # Gradient computation:
    # d(soft_min_i)/d(points_j) = sum_k weights_ik * d(angle_ik)/d(points_j)
    # d(angle_ik)/d(points_j) = 0 if j != k, else -g_i / sqrt(1 - dot^2) projected to tangent
    
    # For each grid point i and config point j:
    # contribution = max_weights[i] * weights[i,j] * (-g_i / sqrt(1 - dots[i,j]^2)) * tangent_projection
    
    # Compute per (i,j) gradient contributions
    # denom: (M, N)
    denom = np.sqrt(np.maximum(1.0 - dots * dots, 1e-18))  # (M, N)
    
    # coeff[i,j] = max_weights[i] * weights[i,j]
    coeff = max_weights[:, np.newaxis] * weights  # (M, N)
    
    # alpha[i,j] = coeff[i,j] / denom[i,j]
    alpha = coeff / denom  # (M, N)
    
    # Term1_j = -sum_i alpha[i,j] * g_i
    # This is: -(alpha.T @ grid_points)  -> shape (N, 3)
    term1 = -(alpha.T @ grid_points)  # (N, 3)
    
    # Term 2: sum_i coeff[i,j] * dots[i,j] / denom[i,j] * p_j
    # = sum_i alpha[i,j] * dots[i,j] * p_j
    # For each j: p_j * sum_i alpha[i,j] * dots[i,j]
    beta = alpha * dots  # (M, N)
    beta_sum = np.sum(beta, axis=0)  # (N,)
    term2 = beta_sum[:, np.newaxis] * points  # (N, 3)
    
    grad = term1 + term2  # (N, 3)
    
    return soft_max_val, grad

def _lloyd_step(points, grid_points, n_iters=3):
    """
    Perform Lloyd's algorithm iterations using grid-based Voronoi.
    Uses vectorized operations for speed.
    """
    for _ in range(n_iters):
        dots = grid_points @ points.T
        angles = np.arccos(np.clip(dots, -1.0, 1.0))
        nearest = np.argmin(angles, axis=1)
        
        new_points = np.zeros_like(points)
        
        # Vectorized: sum grid points per cluster, then divide by count
        for j in range(N):
            mask = (nearest == j)
            count = np.sum(mask)
            if count > 0:
                new_points[j] = np.sum(grid_points[mask], axis=0) / count
            else:
                new_points[j] = points[j]
        
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        new_points = new_points / norms
        
        diff = np.linalg.norm(new_points - points, axis=1)
        points = new_points
        
        if np.max(diff) < 1e-7:
            break
    
    return points

def _exact_local_optimize(points, grid_points, max_iters=200, start_time=0, time_limit=150):
    """
    Final high-precision polish: iteratively move each point to reduce the
    maximum facet circumradius, using exact geometric computations.
    Focuses on the single worst facet to save time.
    """
    pts = points.copy()
    for it in range(max_iters):
        if time.time() - start_time > time_limit - 2.0:
            break
        
        # Compute current covering radius from hull
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Find the single worst facet
        worst_facet = None
        worst_radius = -1
        worst_normal = None
        
        for simplex in hull.simplices:
            verts = pts[simplex]
            a, b, c = verts[0], verts[1], verts[2]
            
            # Compute normal
            normal = np.cross(b - a, c - a)
            norm_val = np.linalg.norm(normal)
            if norm_val < 1e-12:
                continue
            normal = normal / norm_val
            
            # Check orientation
            centroid = np.mean(verts, axis=0)
            if np.dot(normal, centroid) < 0:
                normal = -normal
            
            # Compute angles from vertices to normal
            dots = np.clip(np.dot(verts, normal), -1.0, 1.0)
            angles = np.arccos(dots)
            facet_radius = np.max(angles)
            
            if facet_radius > worst_radius:
                worst_radius = facet_radius
                worst_facet = simplex
                worst_normal = normal
        
        if worst_facet is None or worst_radius < 0.19:
            break
            
        # Move the vertices of the worst facet slightly toward the normal direction
        # This reduces the angular distance from the normal to the vertices
        move_amount = 0.003
        for idx in worst_facet:
            # Move point toward the normal direction (tangent component)
            tangent = worst_normal - np.dot(pts[idx], worst_normal) * pts[idx]
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm > 1e-12:
                pts[idx] = pts[idx] + move_amount * tangent / tangent_norm
        
        # Renormalize
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / norms[:, np.newaxis]
        
    return pts

def construct():
    global start_time, total_time_limit
    start_time = time.time()
    total_time_limit = 143.0  # Leave 7s margin for 150s limit

    # Generate grid on sphere - use moderate density for speed
    n_grid = 20000
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
    for _ in range(2):
        rp = rng.normal(size=(N, 3))
        rp /= np.linalg.norm(rp, axis=1, keepdims=True)
        starts.append(rp)
    
    # Process each start
    for idx, start_pts in enumerate(starts):
        if time.time() - start_time > total_time_limit - 25.0:
            break
        
        points = start_pts.copy()
        
        # Step 1: Lloyd's iterations (fewer to save time for gradient)
        for _ in range(4):
            if time.time() - start_time > total_time_limit - 20.0:
                break
            points = _lloyd_step(points, grid_points, n_iters=3)
        
        # Step 2: Multi-stage gradient optimization with sigma annealing
        if time.time() - start_time < total_time_limit - 15.0:
            sigmas = [0.15, 0.08, 0.04, 0.02, 0.01]
            
            for sigma in sigmas:
                if time.time() - start_time > total_time_limit - 10.0:
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
                        options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-9, 'disp': False}
                    )
                    if res.x is not None:
                        points = res.x.reshape(N, 3)
                        norms = np.linalg.norm(points, axis=1, keepdims=True)
                        norms = np.where(norms < 1e-12, 1.0, norms)
                        points = points / norms
                except Exception:
                    pass
        
        # Step 3: Exact local optimization
        if time.time() - start_time < total_time_limit - 5.0:
            points = _exact_local_optimize(points, grid_points, max_iters=300, start_time=start_time, time_limit=total_time_limit)
        
        # Evaluate exact covering radius
        score = _covering_radius_from_hull(points)
        
        if score < best_score:
            best_score = score
            best_points = points.copy()
        
        if best_score < 0.197:
            break
    
    if best_points is None:
        best_points = spiral_points
        best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
