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

def _facet_based_optimize(points, max_iters=300):
    """
    Iteratively find the worst facet and move its vertices toward the circumcenter.
    This is a direct geometric optimization that reduces the covering radius.
    """
    pts = points.copy()
    
    for it in range(max_iters):
        if time.time() - _time_start > _time_limit - 3.0:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        worst_facet = None
        worst_radius = 0.0
        worst_normal = None
        
        for simplex in hull.simplices:
            verts = pts[simplex]
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
            
            if facet_radius > worst_radius:
                worst_radius = facet_radius
                worst_facet = simplex
                worst_normal = normal
        
        if worst_facet is None or worst_radius < 0.197:
            break
        
        # Move vertices of the worst facet toward the circumcenter
        # The circumcenter is in the direction of the normal from the origin
        # We want to reduce the angular distance from each vertex to the circumcenter
        # Moving a vertex p toward the circumcenter direction reduces the angle
        
        step_size = 0.02
        for j in worst_facet:
            p = pts[j]
            proj = np.dot(p, worst_normal)
            tangent_dir = worst_normal - proj * p
            tangent_norm = np.linalg.norm(tangent_dir)
            if tangent_norm < 1e-12:
                continue
            tangent_dir /= tangent_norm
            
            new_pt = p + step_size * tangent_dir
            new_pt /= np.linalg.norm(new_pt)
            pts[j] = new_pt
    
    return pts

def _penalty_optimize(points, grid_points, target_radius, max_iters=200):
    """
    Penalty-based optimization: maximize sum of squared distances subject to
    minimum distance constraint. Uses augmented Lagrangian approach.
    """
    pts = points.copy()
    penalty = 100.0
    
    for it in range(max_iters):
        if time.time() - _time_start > _time_limit - 5.0:
            break
        
        # Compute pairwise distances
        dots = pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        np.fill_diagonal(angles, np.pi)
        
        # Objective: maximize sum of squared angular distances
        # Penalty: for pairs with angle < target, add penalty * (target - angle)^2
        violations = np.maximum(0, target_radius - angles)
        penalty_term = np.sum(violations ** 2)
        
        # Compute gradient
        # d(sum of angles^2)/d(p_j) for each j
        # d(angle_ij)/d(p_j) = -p_i / sin(angle_ij) + (p_i . p_j) / sin(angle_ij) * p_j
        #                      = -p_i / sin(angle_ij) + cos(angle_ij) / sin(angle_ij) * p_j
        
        sin_angles = np.sin(angles)
        sin_angles = np.maximum(sin_angles, 1e-12)
        
        # Gradient of sum of squared angles
        # d(angle_ij^2)/d(p_j) = 2 * angle_ij * d(angle_ij)/d(p_j)
        # = 2 * angle_ij * (-p_i / sin(angle_ij) + cos(angle_ij) / sin(angle_ij) * p_j)
        
        # For each j: sum over i != j
        # Let's compute this efficiently
        # grad_j = sum_i 2 * angle_ij * (-p_i / sin(angle_ij) + cos(angle_ij) / sin(angle_ij) * p_j)
        # = -2 * sum_i (angle_ij / sin(angle_ij)) * p_i + 2 * sum_i (angle_ij * cos(angle_ij) / sin(angle_ij)) * p_j
        
        cos_angles = np.cos(angles)
        np.fill_diagonal(cos_angles, 1.0)
        
        # term1_j = -2 * sum_i (angle_ij / sin(angle_ij)) * p_i
        coeff1 = angles / sin_angles
        np.fill_diagonal(coeff1, 0.0)
        term1 = -2.0 * (coeff1 @ pts)
        
        # term2_j = 2 * sum_i (angle_ij * cos(angle_ij) / sin(angle_ij)) * p_j
        coeff2 = angles * cos_angles / sin_angles
        np.fill_diagonal(coeff2, 0.0)
        term2 = 2.0 * np.sum(coeff2, axis=0)[:, np.newaxis] * pts
        
        grad_obj = term1 + term2
        
        # Gradient of penalty term
        # d(penalty)/d(p_j) = sum_i 2 * (target - angle_ij) * (-d(angle_ij)/d(p_j)) for violations
        # = -2 * sum_i (target - angle_ij) * (-p_i / sin(angle_ij) + cos(angle_ij) / sin(angle_ij) * p_j) for violations
        # = 2 * sum_i (target - angle_ij) * (p_i / sin(angle_ij) - cos(angle_ij) / sin(angle_ij) * p_j) for violations
        
        viol_mask = violations > 0
        coeff_p1 = np.where(viol_mask, 2.0 * violations / sin_angles, 0.0)
        np.fill_diagonal(coeff_p1, 0.0)
        pterm1 = coeff_p1 @ pts
        
        coeff_p2 = np.where(viol_mask, -2.0 * violations * cos_angles / sin_angles, 0.0)
        np.fill_diagonal(coeff_p2, 0.0)
        pterm2 = np.sum(coeff_p2, axis=0)[:, np.newaxis] * pts
        
        grad_penalty = pterm1 + pterm2
        
        # Total gradient (we want to minimize negative objective + penalty)
        grad = -grad_obj + penalty * grad_penalty
        
        # Take a step
        step_size = 0.01
        new_pts = pts - step_size * grad
        new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
        
        # Check if improvement
        new_dots = new_pts @ new_pts.T
        new_dots = np.clip(new_dots, -1.0, 1.0)
        new_angles = np.arccos(new_dots)
        np.fill_diagonal(new_angles, np.pi)
        
        new_violations = np.maximum(0, target_radius - new_angles)
        new_penalty_term = np.sum(new_violations ** 2)
        
        if new_penalty_term < penalty_term or it % 20 == 0:
            pts = new_pts
        else:
            step_size *= 0.5
        
        # Increase penalty
        if it % 10 == 0:
            penalty *= 1.5
    
    return pts

def construct():
    global _time_start
    _time_start = time.time()
    
    # Generate grid on sphere
    n_grid = 30000
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
        if time.time() - _time_start > _time_limit - 25.0:
            break
        
        points = start_pts.copy()
        
        # Step 1: Lloyd's iterations
        for _ in range(15):
            if time.time() - _time_start > _time_limit - 20.0:
                break
            points = _lloyd_step(points, grid_points, n_iters=3)
        
        # Step 2: Multi-stage gradient optimization with sigma annealing
        if time.time() - _time_start < _time_limit - 15.0:
            sigmas = [0.2, 0.1, 0.05, 0.025, 0.01, 0.005, 0.002]
            
            for sigma in sigmas:
                if time.time() - _time_start > _time_limit - 10.0:
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
                        options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-9, 'disp': False}
                    )
                    if res.x is not None:
                        points = res.x.reshape(N, 3)
                        norms = np.linalg.norm(points, axis=1, keepdims=True)
                        norms = np.where(norms < 1e-12, 1.0, norms)
                        points = points / norms
                except Exception:
                    pass
        
        # Step 3: Penalty-based optimization
        if time.time() - _time_start < _time_limit - 8.0:
            # Estimate current covering radius
            current_radius = _covering_radius_from_hull(points)
            # Target slightly below current to push for improvement
            target = current_radius * 0.98
            points = _penalty_optimize(points, grid_points, target, max_iters=100)
        
        # Step 4: Facet-based exact optimization
        if time.time() - _time_start < _time_limit - 3.0:
            points = _facet_based_optimize(points, max_iters=400)
        
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
