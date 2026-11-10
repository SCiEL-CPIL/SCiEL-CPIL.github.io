import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def _covering_radius(points):
    """
    Calculate the covering radius in radians for points on the unit sphere.
    The covering radius is the maximum circumradius of the convex hull facets.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi # Fallback for degenerate cases
        
    max_radius = 0.0
    for simplex in hull.simplices:
        p0 = points[simplex[0]]
        p1 = points[simplex[1]]
        p2 = points[simplex[2]]
        
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_normal = np.linalg.norm(normal)
        
        if norm_normal < 1e-10:
            continue
            
        d = np.abs(np.dot(normal, p0)) / norm_normal
        d = np.clip(d, 0.0, 1.0)
        alpha = np.arccos(d)
        
        if alpha > max_radius:
            max_radius = alpha
            
    return max_radius

def _compute_voronoi_vertices(points):
    """
    Compute spherical Voronoi vertices (circumcenters of Delaunay triangles).
    Returns array of unit vectors (circumcenters) and their angular distances to nearest point.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None
    
    circumcenters = []
    for simplex in hull.simplices:
        p0 = points[simplex[0]]
        p1 = points[simplex[1]]
        p2 = points[simplex[2]]
        
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_normal = np.linalg.norm(normal)
        
        if norm_normal < 1e-10:
            continue
        
        # Circumcenter is along the normal direction
        # For a triangle on the unit sphere, the circumcenter is at distance 1/sin(alpha) from origin
        # where alpha is the circumradius
        # The circumcenter direction is the normalized normal
        cc = normal / norm_normal
        
        # Check if this is a valid circumcenter (should be on the "outside" of the triangle)
        # The circumcenter should be in the direction of the normal if the triangle is oriented correctly
        # For convex hull facets, the normal points outward
        circumcenters.append(cc)
    
    if not circumcenters:
        return None, None
    
    circumcenters = np.array(circumcenters)
    
    # Compute angular distance from each circumcenter to its nearest generating point
    # dist[i,j] = angle between circumcenters[i] and points[j]
    dot_products = circumcenters @ points.T  # (M, N)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angles = np.arccos(dot_products)  # (M, N)
    
    # Nearest distance for each circumcenter
    nearest_dists = np.min(angles, axis=1)  # (M,)
    
    return circumcenters, nearest_dists

def _smooth_min_surrogate(values, beta=50.0):
    """
    Smooth approximation of the minimum using log-sum-exp.
    min(x) ≈ -(1/beta) * log(sum(exp(-beta * x)))
    This is differentiable and approaches the true min as beta → ∞.
    """
    values = np.asarray(values)
    # For numerical stability, subtract the minimum
    v_min = np.min(values)
    exp_vals = np.exp(-beta * (values - v_min))
    log_sum = np.log(np.sum(exp_vals))
    return -(v_min + log_sum / beta)

def _gradient_descent_voronoi(points, deadline, max_iter=2000, lr=0.01, beta=50.0, beta_anneal=True):
    """
    Gradient descent on the sphere using smooth min-surrogate of Voronoi vertex distances.
    
    Objective: maximize the minimum angular distance from Voronoi vertices to nearest point.
    Equivalent to: minimize -min(distances) = minimize smooth_max(-distances)
    
    We use: f = -(1/beta) * log(sum(exp(-beta * d_i))) as a smooth surrogate for min(d_i)
    We want to maximize f, so we minimize -f.
    """
    points = points.copy()
    current_beta = beta
    best_radius = _covering_radius(points)
    best_points = points.copy()
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute Voronoi vertices and distances
        circumcenters, nearest_dists = _compute_voronoi_vertices(points)
        
        if circumcenters is None or nearest_dists is None or len(nearest_dists) == 0:
            break
        
        # Smooth surrogate for the minimum distance
        # We want to maximize min(nearest_dists)
        # Use smooth max of -nearest_dists: -(1/beta)*log(sum(exp(-beta*(-d_i)))) = -(1/beta)*log(sum(exp(beta*d_i)))
        # Actually, let's use: f = -(1/beta) * log(sum(exp(-beta * d_i))) which approximates min(d_i)
        # We want to maximize f, so gradient of f w.r.t. points gives direction to move
        
        # Compute gradient of the smooth surrogate
        # f = -(1/beta) * log(sum(exp(-beta * d_i)))
        # df/dp = (1/beta) * sum_j [exp(-beta * d_j) * (-beta * dd_j/dp)] / sum_k [exp(-beta * d_k)]
        #        = -sum_j [w_j * dd_j/dp] where w_j = exp(-beta * d_j) / sum_k exp(-beta * d_k)
        
        # For each Voronoi vertex j, dd_j/dp_k is the gradient of the distance from vertex j to point k
        # d_j = arccos(cc_j · p_k) for the nearest point k
        # dd_j/dp_k = -cc_j / sqrt(1 - (cc_j · p_k)^2) for the nearest point
        
        # Compute weights
        exp_vals = np.exp(-current_beta * nearest_dists)
        weights = exp_vals / np.sum(exp_vals)  # (M,)
        
        # Compute gradient for each point
        # For each Voronoi vertex j, find its nearest point index
        nearest_indices = np.argmin(np.arccos(np.clip(circumcenters @ points.T, -1.0, 1.0)), axis=1)
        
        grad = np.zeros_like(points)
        for j in range(len(nearest_dists)):
            k = nearest_indices[j]
            cc = circumcenters[j]
            dot_val = np.dot(cc, points[k])
            dot_val = np.clip(dot_val, -1.0 + 1e-10, 1.0 - 1e-10)
            denom = np.sqrt(1.0 - dot_val * dot_val)
            
            # d_j = arccos(cc · p_k)
            # dd_j/dp_k = -cc / sqrt(1 - (cc · p_k)^2)
            # But we want to maximize d_j, so we move in the direction that increases d_j
            # Increasing d_j means moving p_k away from cc
            # The gradient of d_j w.r.t. p_k is -cc / denom
            # To maximize d_j, we should move in the direction of -gradient = cc / denom
            # But wait: if we move p_k in direction cc, the dot product increases, so arccos decreases
            # So to increase d_j = arccos(cc·p_k), we need to decrease cc·p_k
            # So we move p_k in direction -cc
            
            # Actually: d = arccos(x), dd/dx = -1/sqrt(1-x^2)
            # x = cc · p, dx/dp = cc
            # dd/dp = -cc / sqrt(1 - (cc·p)^2)
            # To maximize d, move in direction -dd/dp = cc / sqrt(1 - (cc·p)^2)
            # But this would increase x = cc·p, which decreases d!
            # So to increase d, we need to move in direction dd/dp = -cc / sqrt(1-(cc·p)^2)
            
            # The force on point k from vertex j is proportional to weight_j * (-cc / denom)
            # This pushes point k away from the circumcenter cc
            
            grad[k] += weights[j] * (-cc / denom)
        
        # Normalize gradient and take step
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norm = np.maximum(grad_norm, 1e-10)
        grad_normalized = grad / grad_norm
        
        # Project onto tangent space of sphere
        # Tangent direction: g - (g·p)p
        tangent_grad = grad_normalized - np.sum(grad_normalized * points, axis=1, keepdims=True) * points
        
        # Take step
        points += lr * tangent_grad
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points /= norms
        
        # Anneal beta
        if beta_anneal and it % 100 == 0:
            current_beta = min(current_beta * 1.05, 200.0)
        
        # Track best
        current_radius = _covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    return best_points

def _repulsive_forces(points, k=1.0):
    """
    Vectorized repulsive forces between all pairs of points.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-12)
    inv_dist_cubed = 1.0 / (dist_sq * np.sqrt(dist_sq))
    np.fill_diagonal(inv_dist_cubed, 0.0)
    forces = np.sum(diff * inv_dist_cubed[:, :, np.newaxis], axis=1)
    return k * forces

def _density_scaled_forces(points, k=1.0, gamma=1.0):
    """
    Repulsive forces scaled by local density.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    nn_dist_sq = np.min(dist_sq, axis=1)
    nn_dist = np.sqrt(nn_dist_sq)
    scale = nn_dist ** gamma
    forces = _repulsive_forces(points, k=1.0)
    forces *= scale[:, np.newaxis]
    return forces

def _repulsive_relaxation(points, steps=2000, alpha=0.05, use_density_scaling=False):
    """
    Relax points using repulsive forces to spread them out.
    """
    points = points.copy()
    for _ in range(steps):
        if use_density_scaling:
            forces = _density_scaled_forces(points, k=1.0, gamma=1.0)
        else:
            forces = _repulsive_forces(points, k=1.0)
        points += alpha * forces
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points /= norms
    return points

def _spherical_to_cartesian(params):
    """
    Convert spherical coordinates (theta, phi) to Cartesian coordinates.
    """
    theta = params[0::2]
    phi = params[1::2]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _cartesian_to_spherical(points):
    """
    Convert Cartesian coordinates to spherical coordinates (theta, phi).
    """
    theta = np.arccos(np.clip(points[:, 2], -1.0, 1.0))
    phi = np.arctan2(points[:, 1], points[:, 0])
    params = np.zeros(2 * len(points))
    params[0::2] = theta
    params[1::2] = phi
    return params

def _local_optimize_nelder_mead(points, deadline, maxiter=500):
    """
    Local optimization using Nelder-Mead on spherical coordinates.
    """
    def objective(params):
        if time.time() > deadline:
            return 1e6
        pts = _spherical_to_cartesian(params)
        if not np.all(np.isfinite(pts)):
            return 1e6
        return _covering_radius(pts)
    
    params0 = _cartesian_to_spherical(points)
    
    try:
        result = minimize(objective, params0, method='Nelder-Mead', 
                          options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-8})
        if result.fun < _covering_radius(points):
            return _spherical_to_cartesian(result.x)
    except Exception:
        pass
        
    return points

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    # 2. Repulsive Relaxation with density scaling
    points = _repulsive_relaxation(points, steps=2000, alpha=0.03, use_density_scaling=True)
    current_radius = _covering_radius(points)
    if current_radius < best_radius:
        best_points = points.copy()
        best_radius = current_radius
    
    # 3. Gradient descent on Voronoi vertices with smooth min-surrogate
    if time.time() < deadline - 40:
        gd_result = _gradient_descent_voronoi(best_points, deadline - 20, max_iter=3000, lr=0.02, beta=30.0)
        gd_radius = _covering_radius(gd_result)
        if gd_radius < best_radius:
            best_points = gd_result
            best_radius = gd_radius
    
    # 4. Nelder-Mead refinement
    if time.time() < deadline - 20:
        optimized = _local_optimize_nelder_mead(best_points, deadline - 5, maxiter=800)
        opt_radius = _covering_radius(optimized)
        if opt_radius < best_radius:
            best_points = optimized
            best_radius = opt_radius
    
    # 5. Multiple restarts with perturbations if time allows
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        np.random.seed(42)
        for restart in range(8):
            if time.time() > deadline - 5:
                break
            
            if restart % 3 == 0:
                noise = np.random.normal(0, 0.05, size=best_points.shape)
            elif restart % 3 == 1:
                noise = np.random.normal(0, 0.15, size=best_points.shape)
            else:
                theta = np.random.uniform(0, 2 * np.pi)
                axis = np.random.normal(size=3)
                axis /= np.linalg.norm(axis)
                K = np.array([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]])
                R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
                noise = best_points @ R.T - best_points
            
            perturbed = best_points + noise
            norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
            perturbed /= norms
            
            # Quick relaxation
            relaxed = _repulsive_relaxation(perturbed, steps=500, alpha=0.03, use_density_scaling=True)
            rad = _covering_radius(relaxed)
            
            if rad < best_radius:
                best_points = relaxed
                best_radius = rad
                
                # Gradient descent refinement
                if time.time() < deadline - 15:
                    gd_res = _gradient_descent_voronoi(best_points, deadline - 5, max_iter=1000, lr=0.015, beta=50.0)
                    gd_rad = _covering_radius(gd_res)
                    if gd_rad < best_radius:
                        best_points = gd_res
                        best_radius = gd_rad
    
    # Final check: ensure points are on the unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points /= norms
    
    return best_points
