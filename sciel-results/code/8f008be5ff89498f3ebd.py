import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _facet_data(points):
    """Compute circumcenters and radii for all facets of the convex hull."""
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
    except Exception:
        return None, None, None
    
    circumcenters = []
    radii = []
    
    for tri in facets:
        p1 = points[tri[0]]
        p2 = points[tri[1]]
        p3 = points[tri[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            circumcenters.append(np.zeros(3))
            radii.append(0.0)
            continue
            
        c = cross / norm_cross
        
        if np.dot(c, p1) < 0:
            c = -c
        
        dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        
        circumcenters.append(c)
        radii.append(dist)
    
    return np.array(circumcenters), np.array(radii), facets

def _compute_covering_radius(points):
    """Compute the covering radius in degrees."""
    circumcenters, radii, facets = _facet_data(points)
    if circumcenters is None:
        return 180.0
    if len(radii) == 0:
        return 0.0
    return np.degrees(np.max(radii))

def _smooth_objective_and_grad(x, temperature=0.01):
    """
    Compute smooth max of facet circumradii and its gradient.
    x is flattened array of shape (3*N,).
    """
    points = x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    points = points / norms
    
    circumcenters, radii, facets = _facet_data(points)
    
    if circumcenters is None:
        return 1.5, np.zeros(3 * N)
    
    if len(radii) == 0:
        return 0.0, np.zeros(3 * N)
    
    # Smooth max
    max_r = np.max(radii)
    exps = np.exp((radii - max_r) / temperature)
    smooth_val = max_r + temperature * np.log(np.sum(exps))
    
    # Weights
    weights = exps / np.sum(exps)
    
    # Gradient computation
    grad_flat = np.zeros(3 * N)
    
    for idx, tri in enumerate(facets):
        w = weights[idx]
        if w < 1e-8:
            continue
        
        r_i = radii[idx]
        c_i = circumcenters[idx]
        
        cos_r = np.dot(c_i, points[tri[0]])
        sin_r = np.sqrt(max(1e-15, 1.0 - cos_r * cos_r))
        
        # Gradient of the distance from vertex p to circumcenter c on the sphere
        # d/dp arccos(c . p) = -c / sin(r) projected onto tangent space
        grad_dir = -c_i / sin_r
        
        for v_idx in tri:
            p = points[v_idx]
            proj = np.dot(grad_dir, p)
            tangent_grad = grad_dir - proj * p
            grad_flat[3*v_idx:3*v_idx+3] += w * tangent_grad
    
    return smooth_val, grad_flat

def _polish_worst_facets(points, max_iters=50, tol=1e-10):
    """
    Iteratively polish the worst facets by moving their vertices towards circumcenters.
    """
    for _ in range(max_iters):
        circumcenters, radii, facets = _facet_data(points)
        if circumcenters is None:
            break
        
        if len(radii) == 0:
            break
        
        max_idx = np.argmax(radii)
        max_r = radii[max_idx]
        
        if max_r < tol:
            break
        
        tri = facets[max_idx]
        c = circumcenters[max_idx]
        
        for v_idx in tri:
            p = points[v_idx]
            direction = c - p
            proj = np.dot(direction, p)
            tangent_dir = direction - proj * p
            norm_td = np.linalg.norm(tangent_dir)
            if norm_td > 1e-12:
                tangent_dir /= norm_td
                # Small step towards circumcenter
                points[v_idx] = points[v_idx] + 0.005 * tangent_dir
                points[v_idx] /= np.linalg.norm(points[v_idx])
    
    return points

def _fibonacci_spiral(N):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial configuration
    points = _fibonacci_spiral(N)
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Phase 1: L-BFGS-B optimization with smooth max surrogate
    temperature = 0.05
    
    from scipy.optimize import minimize
    
    while time.time() < deadline - 40:
        x0 = points.flatten()
        
        def obj_and_grad(x):
            val, grad = _smooth_objective_and_grad(x, temperature)
            return val, grad
        
        try:
            result = minimize(
                obj_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 50, 'gtol': 1e-6, 'maxls': 20}
            )
            
            new_points = result.x.reshape(N, 3)
            new_points = _normalize(new_points)
            
            if not np.all(np.isfinite(new_points)):
                break
            
            new_radius = _compute_covering_radius(new_points)
            
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = new_points.copy()
                points = new_points
                temperature *= 0.8
            else:
                temperature *= 0.7
                if temperature < 0.001:
                    break
        except Exception:
            temperature *= 0.7
            if temperature < 0.001:
                break
    
    # Phase 2: Polish worst facets
    if time.time() < deadline - 10:
        polished = _polish_worst_facets(best_points.copy(), max_iters=50, tol=1e-10)
        polished_radius = _compute_covering_radius(polished)
        if polished_radius < best_radius:
            best_radius = polished_radius
            best_points = polished.copy()
    
    # Phase 3: Final fine-tuning with very small L-BFGS-B steps
    if time.time() < deadline - 5:
        temperature = 0.001
        x0 = best_points.flatten()
        
        def obj_and_grad_final(x):
            val, grad = _smooth_objective_and_grad(x, temperature)
            return val, grad
        
        try:
            result = minimize(
                obj_and_grad_final,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 30, 'gtol': 1e-8, 'maxls': 10}
            )
            
            new_points = result.x.reshape(N, 3)
            new_points = _normalize(new_points)
            
            if np.all(np.isfinite(new_points)):
                new_radius = _compute_covering_radius(new_points)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_points = new_points.copy()
        except Exception:
            pass
    
    # Ensure output is valid
    best_points = _normalize(best_points)
    return best_points
