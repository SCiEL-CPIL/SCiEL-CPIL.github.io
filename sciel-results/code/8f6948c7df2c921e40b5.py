import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130
DEADLINE_MARGIN = 10.0

def _fibonacci_sphere(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y, z])

def _covering_radius_hull(points):
    """
    Compute the covering radius in radians using the convex hull.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    max_angle = 0.0
    
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        e1 = v2 - v1
        e2 = v3 - v1
        normal = np.cross(e1, e2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal = normal / norm
        
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(centroid, normal) < 0:
            normal = -normal
        
        d1 = np.arccos(np.clip(np.dot(normal, v1), -1, 1))
        d2 = np.arccos(np.clip(np.dot(normal, v2), -1, 1))
        d3 = np.arccos(np.clip(np.dot(normal, v3), -1, 1))
        
        facet_radius = max(d1, d2, d3)
        
        # Check if empty cap
        dots = points @ normal
        mask = np.ones(N, dtype=bool)
        mask[simplex] = False
        max_dot_other = np.max(dots[mask])
        
        if max_dot_other > np.cos(facet_radius) + 1e-12:
            continue
            
        max_angle = max(max_angle, facet_radius)
        
    return max_angle

def _lloyd_iteration(points, n_iters=50, deadline=None):
    """
    Lloyd's algorithm on the sphere:
    1. Compute Voronoi cells (via Delaunay / Convex Hull)
    2. Move each point to the centroid (spherical mean) of its cell
    """
    points = points.copy().astype(float)
    for _ in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        circumcenters = []
        facet_vertices = []
        
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            e1 = v2 - v1
            e2 = v3 - v1
            normal = np.cross(e1, e2)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm
            centroid = (v1 + v2 + v3) / 3.0
            if np.dot(centroid, normal) < 0:
                normal = -normal
            
            circumcenters.append(normal)
            facet_vertices.append(simplex)
        
        if len(circumcenters) == 0:
            break
            
        circumcenters = np.array(circumcenters)
        
        new_points = np.zeros_like(points)
        
        for i in range(N):
            mask = [idx for idx, simplex in enumerate(facet_vertices) if i in simplex]
            if len(mask) == 0:
                new_points[i] = points[i]
                continue
            
            cc = circumcenters[mask]
            
            mean = np.sum(cc, axis=0)
            norm = np.linalg.norm(mean)
            if norm < 1e-10:
                new_points[i] = points[i]
            else:
                new_points[i] = mean / norm
        
        points = new_points
        
    return points

def _repulsion_optimize(points, deadline=None):
    """
    Optimize using repulsion-based optimization with penalty method.
    Maximizes sum of squared pairwise distances subject to minimum distance constraints.
    """
    points = points.copy().astype(float)
    
    # Target covering radius in radians (approx 11.316 degrees)
    target_radius = np.radians(11.3165625)
    
    # Penalty weight for violating minimum distance
    penalty_weight = 100.0
    
    # Maximum iterations
    max_iters = 100
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute pairwise distances
        dots = points @ points.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        
        # Find minimum distance
        min_dist = np.min(dists[~np.eye(N, dtype=bool)])
        
        # If minimum distance is less than target, increase penalty
        if min_dist < target_radius:
            penalty_weight *= 1.5
        
        # Objective: maximize sum of squared distances, penalize violations
        def objective(flat_params):
            p = flat_params.reshape(N, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            p = p / norms
            
            dots = p @ p.T
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots)
            
            # Sum of squared distances (maximize)
            sum_sq_dist = np.sum(dists[~np.eye(N, dtype=bool)] ** 2)
            
            # Penalty for violations
            violations = np.maximum(0, target_radius - dists[~np.eye(N, dtype=bool)])
            penalty = penalty_weight * np.sum(violations ** 2)
            
            return -(sum_sq_dist - penalty)
        
        # Gradient
        def objective_and_grad(flat_params):
            p = flat_params.reshape(N, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            p = p / norms
            
            dots = p @ p.T
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots)
            
            # Sum of squared distances
            sum_sq_dist = np.sum(dists[~np.eye(N, dtype=bool)] ** 2)
            
            # Penalty
            violations = np.maximum(0, target_radius - dists[~np.eye(N, dtype=bool)])
            penalty = penalty_weight * np.sum(violations ** 2)
            
            obj = -(sum_sq_dist - penalty)
            
            # Gradient computation
            # d(sum_sq_dist)/d(p_i) = 2 * sum_j (d_ij / sqrt(1-d_ij^2)) * (p_j - (p_i^T p_j) p_i)
            # This is complex, so we use numerical gradient for simplicity
            eps = 1e-7
            grad = np.zeros_like(flat_params)
            for k in range(N * 3):
                orig = flat_params[k]
                flat_params[k] = orig + eps
                f_plus = objective(flat_params)
                flat_params[k] = orig - eps
                f_minus = objective(flat_params)
                flat_params[k] = orig
                grad[k] = (f_plus - f_minus) / (2 * eps)
            
            return obj, grad
        
        try:
            res = minimize(
                objective_and_grad,
                points.ravel(),
                method='L-BFGS-B',
                options={'maxiter': 20, 'gtol': 1e-4, 'maxfun': 50}
            )
            if res.fun < objective(points.ravel()):
                new_points = res.x.reshape(N, 3)
                norms = np.linalg.norm(new_points, axis=1, keepdims=True)
                new_points = new_points / norms
                points = new_points
        except Exception:
            pass
        
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    inits = []
    inits.append(_fibonacci_sphere(N, offset=0.5))
    inits.append(_fibonacci_sphere(N, offset=0.1))
    
    rng = np.random.default_rng(42)
    for _ in range(5):
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)
    
    best_points = None
    best_radius = np.pi
    
    # Evaluate initial configurations
    for pts in inits:
        r = _covering_radius_hull(pts)
        if r < best_radius:
            best_radius = r
            best_points = pts.copy()
        if time.time() > deadline - 10:
            break
            
    # Optimize using Lloyd's algorithm
    if best_points is not None and time.time() < deadline - 5:
        lloyd_pts = _lloyd_iteration(best_points, n_iters=100, deadline=deadline - 5)
        r = _covering_radius_hull(lloyd_pts)
        if r < best_radius:
            best_radius = r
            best_points = lloyd_pts.copy()
            
    # Try repulsion optimization
    if best_points is not None and time.time() < deadline - 10:
        repulsion_pts = _repulsion_optimize(best_points, deadline=deadline - 5)
        r = _covering_radius_hull(repulsion_pts)
        if r < best_radius:
            best_radius = r
            best_points = repulsion_pts.copy()
            
    # Try optimizing from other initial configs
    for pts in inits:
        if time.time() > deadline - 10:
            break
        if np.allclose(pts, best_points):
            continue
            
        lloyd_pts = _lloyd_iteration(pts, n_iters=50, deadline=deadline - 5)
        r = _covering_radius_hull(lloyd_pts)
        if r < best_radius:
            best_radius = r
            best_points = lloyd_pts.copy()
            
            if time.time() < deadline - 10:
                repulsion_pts = _repulsion_optimize(best_points, deadline=deadline - 5)
                r = _covering_radius_hull(repulsion_pts)
                if r < best_radius:
                    best_radius = r
                    best_points = repulsion_pts.copy()
                    
    # Final refinement
    if best_points is not None and time.time() < deadline - 2:
        lloyd_pts = _lloyd_iteration(best_points, n_iters=20, deadline=deadline - 1)
        r = _covering_radius_hull(lloyd_pts)
        if r < best_radius:
            best_points = lloyd_pts.copy()
            
        if time.time() < deadline - 1:
            repulsion_pts = _repulsion_optimize(best_points, deadline=deadline - 0.5)
            r = _covering_radius_hull(repulsion_pts)
            if r < best_radius:
                best_points = repulsion_pts.copy()

    if best_points is None:
        best_points = inits[0]
        
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(pts.shape)
    r = _covering_radius_hull(pts)
    print(f"Covering radius: {np.degrees(r):.6f} degrees")
