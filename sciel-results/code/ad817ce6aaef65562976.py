import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import time
import warnings

N = 130

def _covering_radius_from_hull(points):
    """
    Compute the exact covering radius from the convex hull.
    The covering radius is the maximum circumradius of the spherical caps
    corresponding to the facets of the convex hull.
    
    For a facet with normal n (pointing outward from origin), the cap center is n.
    The angular radius is the maximum angle between n and any vertex of that facet.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # fallback
    
    max_radius = 0.0
    for simplex in hull.simplices:
        # Get the vertices of this facet
        verts = points[simplex]
        
        # The normal of the facet (pointing away from origin)
        # For a triangle with vertices a, b, c on unit sphere:
        # normal = cross(b-a, c-a) normalized
        a, b, c = verts[0], verts[1], verts[2]
        normal = np.cross(b - a, c - a)
        norm_val = np.linalg.norm(normal)
        
        if norm_val < 1e-12:
            continue
            
        normal = normal / norm_val
        
        # Ensure normal points away from origin
        # The centroid of the facet should be in the direction of the normal
        centroid = np.mean(verts, axis=0)
        if np.dot(normal, centroid) < 0:
            normal = -normal
            norm_val = -norm_val
        
        # The cap center is the normal direction
        # The angular radius is the maximum angle between normal and each vertex
        # cos(angle) = dot(normal, vertex)
        dots = np.clip(np.dot(verts, normal), -1.0, 1.0)
        angles = np.arccos(dots)
        
        facet_radius = np.max(angles)
        max_radius = max(max_radius, facet_radius)
    
    return max_radius

def _soft_covering_radius(points, grid_points, sigma=0.05):
    """
    Soft approximation of the covering radius for smooth optimization.
    Uses a soft-min over the grid to create a differentiable objective.
    """
    # Compute angular distances from each grid point to each configuration point
    # dot product gives cos(angle)
    dots = grid_points @ points.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (M, N)
    
    # For each grid point, find the minimum angle (nearest neighbor)
    min_angles = np.min(angles, axis=1)  # (M,)
    
    # Soft max to make it smooth
    # Use log-sum-exp trick for numerical stability
    # soft_max(x) = log(sum(exp(x/sigma))) * sigma
    scaled = min_angles / sigma
    max_scaled = np.max(scaled)
    soft_max_val = sigma * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    
    return soft_max_val

def _gradient_of_soft_covering_radius(points, grid_points, sigma=0.05):
    """
    Compute gradient of the soft covering radius with respect to points.
    """
    M = grid_points.shape[0]
    N_pts = points.shape[0]
    
    # Compute angles
    dots = grid_points @ points.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # (M, N)
    
    # Find which point is nearest for each grid point
    nearest_idx = np.argmin(angles, axis=1)  # (M,)
    min_angles = angles[np.arange(M), nearest_idx]  # (M,)
    
    # Soft max weights
    scaled = min_angles / sigma
    max_scaled = np.max(scaled)
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / np.sum(exp_scaled)  # (M,)
    
    # Gradient with respect to points
    # d(soft_max)/d(point_j) = sum_i w_i * d(min_angle_i)/d(point_j)
    # For the nearest point j to grid point i:
    # d(angle)/d(point_j) where angle = arccos(dot(grid_i, point_j))
    # = -dot(grid_i, point_j) / sqrt(1 - dot(grid_i, point_j)^2) * (grid_i - dot(grid_i, point_j)*point_j) / ||...||
    # Actually, d(arccos(x))/dx = -1/sqrt(1-x^2)
    # d(dot(g, p))/dp = g
    # So d(angle)/dp = -g / sqrt(1 - dot(g,p)^2) ... but need to project onto sphere
    
    grad = np.zeros_like(points)
    
    for i in range(M):
        w = weights[i]
        if w < 1e-10:
            continue
        
        j = nearest_idx[i]
        g = grid_points[i]
        p = points[j]
        
        dot_val = np.dot(g, p)
        dot_val = np.clip(dot_val, -1.0 + 1e-9, 1.0 - 1e-9)
        
        # d(angle)/dp = -g / sqrt(1 - dot^2)
        # But we need to project this gradient onto the tangent plane of the sphere at p
        # The tangent plane constraint: grad_tangent = grad - (grad . p) * p
        grad_raw = -g / np.sqrt(1.0 - dot_val * dot_val)
        
        # Project onto tangent plane
        grad_tangent = grad_raw - np.dot(grad_raw, p) * p
        
        grad[j] += w * grad_tangent
    
    # Normalize gradient to have reasonable magnitude
    grad_norm = np.linalg.norm(grad)
    if grad_norm > 1e-10:
        grad = grad / grad_norm * np.max(np.abs(grad))
    
    return grad

def _lloyd_step(points, grid_points, n_iters=5):
    """
    Perform Lloyd's algorithm iterations using grid-based Voronoi.
    """
    for _ in range(n_iters):
        # Assign each grid point to nearest configuration point
        dots = grid_points @ points.T
        nearest = np.argmin(np.arccos(np.clip(dots, -1.0, 1.0)), axis=1)
        
        # For each configuration point, compute the centroid of its Voronoi region
        new_points = np.zeros_like(points)
        counts = np.zeros(N)
        
        for j in range(N):
            mask = (nearest == j)
            if np.sum(mask) > 0:
                new_points[j] = np.mean(grid_points[mask], axis=0)
                counts[j] = np.sum(mask)
            else:
                # If no points assigned, keep the old position
                new_points[j] = points[j]
        
        # Normalize to unit sphere
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        new_points = new_points / norms
        
        # Check for convergence
        diff = np.linalg.norm(new_points - points, axis=1)
        points = new_points
        
        if np.max(diff) < 1e-6:
            break
    
    return points

def construct():
    start_time = time.time()
    total_time_limit = 140.0  # Leave 10s margin for 150s limit
    
    # Generate a dense grid on the sphere for evaluation
    n_grid = 20000
    i = np.arange(n_grid) + 0.5
    z_grid = 1.0 - 2.0 * i / n_grid
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    grid_points = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    best_score = np.inf
    best_points = None
    
    # Generate multiple starting configurations
    rng = np.random.default_rng(42)
    
    starts = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    spiral_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    starts.append(spiral_points)
    
    # 2. Random points
    for _ in range(10):
        rp = rng.normal(size=(N, 3))
        rp /= np.linalg.norm(rp, axis=1, keepdims=True)
        starts.append(rp)
    
    # 3. Icosahedron-based with jitter
    t = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ])
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Create 130 points by jittering and duplicating the icosahedron vertices
    ico_jittered = ico_verts + rng.normal(scale=0.1, size=(12, 3))
    ico_jittered /= np.linalg.norm(ico_jittered, axis=1, keepdims=True)
    
    # Tile to get 130 points
    ico_tiled = np.tile(ico_jittered, (11, 1))[:N]
    starts.append(ico_tiled)
    
    # Process each start
    for idx, start_pts in enumerate(starts):
        if time.time() - start_time > total_time_limit - 20.0:
            break
        
        points = start_pts.copy()
        
        # Step 1: Lloyd's algorithm iterations
        remaining_time = total_time_limit - (time.time() - start_time)
        lloyd_iters = min(20, max(5, int(remaining_time / 5)))
        
        for _ in range(lloyd_iters):
            if time.time() - start_time > total_time_limit - 15.0:
                break
            points = _lloyd_step(points, grid_points, n_iters=3)
        
        # Step 2: Gradient descent with soft covering radius
        if time.time() - start_time < total_time_limit - 10.0:
            # Use L-BFGS-B with the soft covering radius
            def objective_and_grad(x_flat):
                pts = x_flat.reshape(N, 3)
                # Normalize
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                norms = np.where(norms < 1e-12, 1.0, norms)
                pts = pts / norms
                
                val = _soft_covering_radius(pts, grid_points, sigma=0.03)
                grad = _gradient_of_soft_covering_radius(pts, grid_points, sigma=0.03)
                
                return val, grad.flatten()
            
            x0 = points.flatten()
            
            try:
                res = minimize(
                    objective_and_grad,
                    x0,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5, 'disp': False}
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
        
        # Early exit if we're close to the known best
        if best_score < 0.1975:  # ~11.3 degrees
            break
    
    # Final fallback
    if best_points is None:
        best_points = spiral_points
        best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
