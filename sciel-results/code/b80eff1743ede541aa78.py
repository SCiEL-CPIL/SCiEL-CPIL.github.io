import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _compute_facet_data(points):
    """
    Computes circumradius and circumcenter for each facet of the convex hull.
    Returns radii, centers, simplices.
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return np.array([]), np.array([]), np.array([])
    
    if len(simplices) == 0:
        return np.array([]), np.array([]), np.array([])
        
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    # Compute normals
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    valid = norms > 1e-10
    if not np.all(valid):
        radii = np.full(len(simplices), np.pi)
        centers = np.zeros((len(simplices), 3))
        centers[~valid] = [0, 0, 1]
        return radii, centers, simplices
        
    u = n / norms[:, np.newaxis]
    
    # Orient normals outward (away from origin)
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius: angle between circumcenter (u) and vertex a
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    return radii, u, simplices

def _compute_covering_radius(points):
    """Compute covering radius as max facet circumradius."""
    radii, _, _ = _compute_facet_data(points)
    if len(radii) == 0:
        return np.pi / 2.0
    return np.max(radii)

def _lloyd_iteration(points):
    """One Lloyd iteration: move each point to centroid of its Voronoi cell."""
    try:
        from scipy.spatial import SphericalVoronoi
        vor = SphericalVoronoi(points)
        new_points = np.zeros_like(points)
        for i in range(N):
            region = vor.regions[i]
            if not region:
                new_points[i] = points[i]
                continue
            verts = vor.vertices[region]
            c = np.sum(verts, axis=0)
            norm = np.linalg.norm(c)
            if norm < 1e-10:
                new_points[i] = points[i]
            else:
                new_points[i] = c / norm
        return new_points
    except Exception:
        return points

def _facet_force_step(points, step_size):
    """
    Move points toward circumcenters of large-circumradius facets.
    Force is weighted by how much larger each facet's radius is vs mean.
    """
    radii, centers, simplices = _compute_facet_data(points)
    if len(radii) == 0:
        return points
    
    mean_r = np.mean(radii)
    weights = np.maximum(radii - mean_r, 0.0)
    w_sum = np.sum(weights)
    if w_sum < 1e-15:
        return points
    weights = weights / w_sum
    
    forces = np.zeros_like(points)
    for f in range(len(radii)):
        w = weights[f]
        if w < 1e-15:
            continue
        c = centers[f]
        for idx in simplices[f]:
            d = c - points[idx]
            d_tangent = d - np.dot(d, points[idx]) * points[idx]
            forces[idx] += w * d_tangent
    
    force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
    force_norms[force_norms == 0] = 1.0
    unit_forces = forces / force_norms
    
    new_points = points + step_size * unit_forces
    norms = np.linalg.norm(new_points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return new_points / norms

def _polish_worst_facets(points, max_facets=30, max_iter=200):
    """
    Polish by minimizing max facet radius using Nelder-Mead on vertices
    of the worst facets.
    """
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    for iteration in range(max_iter):
        radii, centers, simplices = _compute_facet_data(best_points)
        if len(radii) == 0:
            break
        
        # Get indices of vertices in the worst facets
        top_k = min(max_facets, len(radii))
        worst_idx = np.argsort(radii)[-top_k:]
        worst_vertices = set()
        for fi in worst_idx:
            for v in simplices[fi]:
                worst_vertices.add(v)
        
        if len(worst_vertices) == 0:
            break
        
        worst_vertices = sorted(worst_vertices)
        n_opt = len(worst_vertices)
        
        def objective(flat):
            pts = best_points.copy()
            for i, vi in enumerate(worst_vertices):
                pts[vi] = flat[3*i:3*i+3]
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            pts = pts / norms
            return _compute_covering_radius(pts)
        
        x0 = np.array([best_points[vi].flatten() for vi in worst_vertices]).flatten()
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if res.fun < best_radius:
                best_radius = res.fun
                new_pts = best_points.copy()
                for i, vi in enumerate(worst_vertices):
                    new_pts[vi] = res.x[3*i:3*i+3]
                norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                best_points = new_pts / norms
            else:
                break
                
        except Exception:
            break
    
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 35s safety margin
    
    # Fibonacci initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Phase 1: Lloyd's algorithm to get a good starting point
    for k in range(20):
        if time.time() > deadline - 60.0:
            break
        points = _lloyd_iteration(points)
        current_radius = _compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
    
    # Phase 2: Facet force balancing
    step_size = 0.05
    for it in range(100):
        if time.time() > deadline - 40.0:
            break
        points = _facet_force_step(points, step_size)
        current_radius = _compute_covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            step_size = min(step_size * 1.2, 0.12)
        else:
            step_size *= 0.7
            if step_size < 1e-5:
                break
    
    # Phase 3: Simulated annealing with restarts
    np.random.seed(42)
    T = 0.08
    cool_rate = 0.997
    no_improve = 0
    
    for step in range(1500):
        if time.time() > deadline - 20.0:
            break
        
        # Random perturbation
        noise_scale = T * 0.1
        perturbed = points + np.random.randn(N, 3) * noise_scale
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        perturbed = perturbed / norms
        
        current_radius = _compute_covering_radius(perturbed)
        
        # Accept if better, or with probability exp(-delta/T)
        delta = current_radius - _compute_covering_radius(points)
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            points = perturbed
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        T *= cool_rate
        
        if no_improve > 100:
            # Restart with perturbation from best
            points = best_points + np.random.randn(N, 3) * 0.1
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            points = points / norms
            T = 0.05
            no_improve = 0
        
        # Periodic facet force step
        if step % 50 == 0:
            points = _facet_force_step(points, 0.03)
    
    # Phase 4: Final polishing on worst facets
    if time.time() < deadline - 5.0:
        polished, polished_radius = _polish_worst_facets(best_points, max_facets=25, max_iter=100)
        if polished_radius < best_radius:
            best_radius = polished_radius
            best_points = polished
    
    # Ensure unit norm and return
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_points = best_points / norms
    
    return best_points
