import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_radius = 0.0
        
        for tri in facets:
            p1 = points[tri[0]]
            p2 = points[tri[1]]
            p3 = points[tri[2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                continue
                
            c = cross / norm_cross
            
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            dist = np.arccos(dot_val)
            
            if dist > np.pi / 2:
                dist = np.pi - dist
                
            if dist > max_radius:
                max_radius = dist
                
    except Exception:
        return 180.0
        
    return np.degrees(max_radius)

def _get_circumcenters_and_radii(points, hull):
    """
    Compute circumcenters and radii for all hull facets.
    Returns centers (M, 3), radii (M,), and facets (M, 3).
    """
    facets = hull.simplices
    M = len(facets)
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    for k in range(M):
        tri = facets[k]
        p1 = points[tri[0]]
        p2 = points[tri[1]]
        p3 = points[tri[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            centers[k] = p1 
            radii[k] = 0.0
            continue
            
        c = cross / norm_cross
        
        # Ensure c points in the direction of the "empty cap" center relative to the facet normal
        centroid = (p1 + p2 + p3) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c
            
        centers[k] = c
        dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        
        if dist > np.pi / 2:
            dist = np.pi - dist
            
        radii[k] = dist
        
    return centers, radii, facets

def _soft_max_radius(points, beta=50.0):
    """
    Approximate the covering radius using a soft-max of pairwise distances.
    This is differentiable and faster than convex hull for intermediate steps.
    """
    # Compute all pairwise angular distances
    # diff[i,j] = points[i] - points[j]
    # dist_sq[i,j] = 2 - 2 * dot(points[i], points[j])
    dots = points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    # For each point i, find the distance to the nearest other point
    # Set diagonal to infinity so it's ignored
    np.fill_diagonal(dists, np.inf)
    
    # We want to maximize the minimum distance? No, we want to minimize the maximum of (distance to nearest).
    # Let d_i = min_{j != i} dist(i, j).
    # We want to minimize max_i d_i.
    # This is equivalent to minimizing the covering radius roughly (Voronoi vertices are further, but this is a good proxy).
    # Actually, the covering radius is the max over the sphere, which is >= max over points.
    # Using soft-max over the "worst" points is a good differentiable proxy.
    
    # min_dist[i] = min_j dists[i, j]
    min_dists = np.min(dists, axis=1)
    
    # Soft max of min_dists
    # log-sum-exp trick
    max_val = np.max(min_dists)
    exp_vals = np.exp(beta * (min_dists - max_val))
    soft_max_val = max_val + np.log(np.sum(exp_vals)) / beta
    
    return soft_max_val

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    points += rng.normal(0, 0.01, points.shape)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Optimization parameters
    step_size = 0.05
    min_step_size = 1e-6
    no_improve_count = 0
    max_no_improve = 20
    beta = 50.0
    
    while time.time() < deadline:
        
        # Use soft-max gradient for fast updates
        # We approximate the gradient of the soft-max radius.
        # However, calculating the exact gradient of the soft-max of mins is complex.
        # Instead, we use the "move towards circumcenter" strategy from inspiration 1,
        # but we use it more aggressively and frequently.
        
        # Compute Hull and circumcenters
        try:
            hull = ConvexHull(points)
            centers, radii, facets = _get_circumcenters_and_radii(points, hull)
        except Exception:
            # If hull fails, try a random kick
            kick = rng.normal(0, 0.01, points.shape)
            points += kick
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue

        # For each point, find the adjacent triangle with the maximum radius
        max_move = np.zeros_like(points)
        
        # Initialize max radius per point to 0
        point_max_radius = np.zeros(N)
        point_max_center = np.zeros((N, 3))
        
        for k in range(len(facets)):
            r_k = radii[k]
            if r_k <= 0: continue
            
            c_k = centers[k]
            tri = facets[k]
            for idx in tri:
                if r_k > point_max_radius[idx]:
                    point_max_radius[idx] = r_k
                    point_max_center[idx] = c_k
        
        # Move points towards their max center
        for i in range(N):
            if point_max_radius[i] > 0:
                v = point_max_center[i] - points[i]
                # Project v onto tangent plane at points[i]
                dot_vp = np.dot(v, points[i])
                tangent_v = v - dot_vp * points[i]
                norm_tv = np.linalg.norm(tangent_v)
                if norm_tv > 1e-12:
                    max_move[i] = tangent_v / norm_tv
                    # Scale by radius to prioritize large holes
                    max_move[i] *= point_max_radius[i]
        
        # Update points
        new_points = points + step_size * max_move
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        # Evaluate using the true metric (ConvexHull) but less frequently?
        # Or use soft-max as a quick check?
        # Given N=130, ConvexHull is fast enough. Let's use it for accuracy.
        current_radius = _compute_covering_radius(new_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points
            no_improve_count = 0
            # Increase step size slightly if we are making progress
            step_size = min(step_size * 1.05, 0.1)
        else:
            # If worse, revert and decrease step size
            step_size = max(step_size * 0.9, min_step_size)
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Stagnation: try a random kick
                kick = rng.normal(0, 0.05, points.shape)
                points += kick
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                step_size = 0.01
                no_improve_count = 0
                current_radius = _compute_covering_radius(points)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = points.copy()

    return best_points
