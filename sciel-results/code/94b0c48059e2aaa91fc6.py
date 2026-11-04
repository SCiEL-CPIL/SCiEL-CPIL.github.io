import numpy as np
from scipy.spatial import SphericalVoronoi, ConvexHull
from scipy.optimize import minimize
import time

N = 130
DEADLINE = 135.0  # safety margin under 150s kill

def _compute_covering_radius(points):
    """
    Computes the covering radius (in radians) for points on the unit sphere.
    Uses the convex hull facets: the covering radius equals the maximum
    circumradius of the Delaunay triangles (facets of the convex hull).
    """
    try:
        # Ensure points are on the unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        points = points / norms
        
        hull = ConvexHull(points)
        simplices = hull.simplices
        F = len(simplices)
        
        max_circ = 0.0
        for simp in simplices:
            v1, v2, v3 = points[simp[0]], points[simp[1]], points[simp[2]]
            # Circumcenter of triangle on sphere
            # c is parallel to cross(v1-v2, v2-v3)
            d1 = v1 - v2
            d2 = v2 - v3
            c_dir = np.cross(d1, d2)
            c_norm = np.linalg.norm(c_dir)
            if c_norm < 1e-14:
                # Degenerate triangle
                circ = np.arccos(np.clip(np.max([np.dot(v1,v2), np.dot(v1,v3), np.dot(v2,v3)]), -1.0, 1.0))
            else:
                c_dir = c_dir / c_norm
                # Choose direction pointing outward (away from origin)
                centroid = (v1 + v2 + v3) / 3.0
                if np.dot(c_dir, centroid) < 0:
                    c_dir = -c_dir
                cos_r = np.dot(c_dir, v1)
                cos_r = np.clip(cos_r, -1.0, 1.0)
                circ = np.arccos(cos_r)
            
            if circ > max_circ:
                max_circ = circ
        
        return max_circ
    except Exception:
        return np.pi / 2.0

def _compute_facet_data(points):
    """
    For each Delaunay triangle (facet of convex hull), compute:
    - circumradius (angular distance from circumcenter to each vertex)
    - circumcenter (point on sphere equidistant to the 3 vertices)
    Returns:
      - radii: array of shape (F,)
      - centers: array of shape (F, 3)
      - simplices: array of shape (F, 3) with vertex indices
    """
    try:
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        points = points / norms
        
        hull = ConvexHull(points)
        simplices = hull.simplices
        F = len(simplices)
        radii = np.zeros(F)
        centers = np.zeros((F, 3))
        
        for i, simp in enumerate(simplices):
            v1, v2, v3 = points[simp[0]], points[simp[1]], points[simp[2]]
            d1 = v1 - v2
            d2 = v2 - v3
            c_dir = np.cross(d1, d2)
            c_norm = np.linalg.norm(c_dir)
            if c_norm < 1e-14:
                radii[i] = np.arccos(np.clip(np.max([np.dot(v1,v2), np.dot(v1,v3), np.dot(v2,v3)]), -1.0, 1.0))
                centers[i] = v1
                continue
            c_dir = c_dir / c_norm
            centroid = (v1 + v2 + v3) / 3.0
            if np.dot(c_dir, centroid) < 0:
                c_dir = -c_dir
            cos_r = np.dot(c_dir, v1)
            cos_r = np.clip(cos_r, -1.0, 1.0)
            radii[i] = np.arccos(cos_r)
            centers[i] = c_dir
        
        return radii, centers, simplices
    except Exception:
        return None, None, None

def _soft_max(x, beta=50.0):
    """Smooth approximation of max(x) using log-sum-exp."""
    x_max = np.max(x)
    if x_max > 1e10:
        return x_max
    return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta

def _smooth_objective(points, beta=50.0):
    """
    Smooth surrogate for the covering radius using log-sum-exp over facet circumradii.
    """
    try:
        radii, centers, simplices = _compute_facet_data(points)
        if radii is None:
            return np.pi / 2.0
        return _soft_max(radii, beta)
    except Exception:
        return np.pi / 2.0

def _smooth_objective_and_grad(points, beta=50.0):
    """
    Compute value and gradient of the smooth covering radius surrogate.
    Uses finite differences for the gradient.
    """
    f0 = _smooth_objective(points, beta)
    eps = 1e-6
    grad = np.zeros_like(points)
    
    for i in range(N):
        for j in range(3):
            orig = points[i, j]
            
            # +eps
            points[i, j] = orig + eps
            norm = np.linalg.norm(points[i])
            if norm > 1e-15:
                points[i] = points[i] / norm
            f_plus = _smooth_objective(points, beta)
            
            # -eps
            points[i, j] = orig - eps
            norm = np.linalg.norm(points[i])
            if norm > 1e-15:
                points[i] = points[i] / norm
            f_minus = _smooth_objective(points, beta)
            
            # Restore
            points[i, j] = orig
            norm = np.linalg.norm(points[i])
            if norm > 1e-15:
                points[i] = points[i] / norm
            
            grad[i, j] = (f_plus - f_minus) / (2 * eps)
    
    # Project gradient onto tangent space
    for i in range(N):
        grad[i] = grad[i] - np.dot(grad[i], points[i]) * points[i]
    
    return f0, grad

def _lloyd_step(points):
    """
    One Lloyd's algorithm step: move each point to the centroid of its
    Voronoi cell (spherical mean).
    """
    try:
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

def _facet_push_step(points, step_size=0.05):
    """
    Balancing force: move points toward circumcenters of large-circumradius facets.
    """
    try:
        radii, centers, simplices = _compute_facet_data(points)
        if radii is None:
            return points
        
        F = len(radii)
        mean_r = np.mean(radii)
        weights = np.maximum(radii - mean_r, 0.0)
        w_sum = np.sum(weights)
        if w_sum < 1e-15:
            return points
        weights = weights / w_sum
        
        forces = np.zeros_like(points)
        for f in range(F):
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
        new_points = new_points / norms
        return new_points
    except Exception:
        return points

def _nelder_mead_polish(points, deadline):
    """
    Final polish with Nelder-Mead on the exact covering radius.
    """
    def objective(flat_points):
        if time.time() > deadline:
            return 10.0
        pts = flat_points.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
        return _compute_covering_radius(pts)
    
    initial_flat = points.flatten()
    try:
        res = minimize(objective, initial_flat, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if res.fun < 10.0:
            best_flat = res.x
            norms = np.linalg.norm(best_flat.reshape(N, 3), axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return best_flat.reshape(N, 3) / norms
    except Exception:
        pass
    return points

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    best_points = points.copy()
    best_radius = _compute_covering_radius(points)

    # 2. Lloyd's Algorithm (Spherical K-Means)
    for k in range(40):
        if time.time() > deadline - 50.0:
            break
        try:
            new_points = _lloyd_step(points)
            diff = np.linalg.norm(new_points - points, axis=1)
            if np.max(diff) < 1e-7:
                break
            points = new_points
            current_radius = _compute_covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
        except Exception:
            break

    # 3. Facet balancing (equalize circumradii)
    step_size = 0.05
    for it in range(400):
        if time.time() > deadline - 40.0:
            break
        try:
            new_points = _facet_push_step(points, step_size)
            new_radius = _compute_covering_radius(new_points)
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = new_points.copy()
                points = new_points
                step_size = min(step_size * 1.2, 0.12)
            else:
                step_size *= 0.6
                if step_size < 1e-6:
                    break
        except Exception:
            break

    # 4. Smooth gradient descent
    if time.time() < deadline - 25.0:
        beta = 30.0
        lr = 0.02
        for it in range(80):
            if time.time() > deadline - 20.0:
                break
            try:
                f0, grad = _smooth_objective_and_grad(points, beta)
                grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
                grad_norm[grad_norm == 0] = 1.0
                grad = grad / grad_norm
                
                new_points = points - lr * grad
                norms = np.linalg.norm(new_points, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                new_points = new_points / norms
                
                new_radius = _compute_covering_radius(new_points)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_points = new_points.copy()
                    points = new_points
                    lr = min(lr * 1.05, 0.08)
                else:
                    lr *= 0.7
                    if lr < 1e-7:
                        break
            except Exception:
                break

    # 5. Final Nelder-Mead polish
    if time.time() < deadline - 5.0:
        points = _nelder_mead_polish(points, deadline - 2.0)
        final_radius = _compute_covering_radius(points)
        if final_radius < best_radius:
            best_radius = final_radius
            best_points = points.copy()

    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_points = best_points / norms

    return best_points
