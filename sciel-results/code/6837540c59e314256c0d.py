import numpy as np
from scipy.spatial import SphericalVoronoi, ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _compute_covering_radius(points):
    """
    Computes the covering radius for points on the unit sphere.
    The covering radius is the maximum distance from any point on the sphere
    to the nearest point in 'points'.
    This is equal to the maximum circumradius of the spherical Voronoi cells.
    """
    try:
        vor = SphericalVoronoi(points)
        vertices = vor.vertices
        dots = vertices @ points.T
        dots = np.clip(dots, -1.0, 1.0)
        min_dots = np.max(dots, axis=1)
        min_dists = np.arccos(min_dots)
        return np.max(min_dists)
    except Exception:
        return np.pi / 2.0

def _compute_facet_circumradii_and_centers(points):
    """
    For each Delaunay triangle (facet of convex hull), compute:
    - the circumradius (angular distance from circumcenter to each vertex)
    - the circumcenter (point on sphere equidistant to the 3 vertices)
    Returns:
      - radii: array of shape (F,) with circumradius for each facet
      - centers: array of shape (F, 3) with circumcenters
      - simplices: array of shape (F, 3) with vertex indices
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
        F = len(simplices)
        radii = np.zeros(F)
        centers = np.zeros((F, 3))
        
        for i, simp in enumerate(simplices):
            v1, v2, v3 = points[simp[0]], points[simp[1]], points[simp[2]]
            # Compute circumcenter of triangle on sphere using linear algebra
            # We want c on unit sphere such that c·v1 = c·v2 = c·v3
            # This means c is orthogonal to (v1-v2) and (v2-v3)
            # So c is parallel to cross(v1-v2, v2-v3)
            d1 = v1 - v2
            d2 = v2 - v3
            c_dir = np.cross(d1, d2)
            c_norm = np.linalg.norm(c_dir)
            if c_norm < 1e-14:
                radii[i] = np.pi / 2.0
                centers[i] = v1
                continue
            c_dir = c_dir / c_norm
            
            # Choose the direction that points "outward" (toward the interior of the Voronoi cell)
            # The correct circumcenter should be on the same side as the origin relative to the triangle plane
            # Or equivalently, we want the circumcenter that maximizes the minimum distance to vertices
            dots_pos = np.array([np.dot(c_dir, v1), np.dot(c_dir, v2), np.dot(c_dir, v3)])
            dots_neg = -dots_pos
            
            # The correct direction is the one where the circumcenter is "outside" the triangle
            # For a valid Delaunay triangulation, the circumcenter of each facet should be
            # on the opposite side of the facet from the origin (for convex hull)
            # Let's check which direction gives a valid circumcenter by verifying
            # that the circumcenter is equidistant to all three vertices
            var_pos = np.var(dots_pos)
            var_neg = np.var(dots_neg)
            
            # Both should be nearly equal since c_dir is orthogonal to d1 and d2
            # Choose the direction that points away from the origin (outward normal)
            # For a convex hull, the outward normal points away from the interior
            # The interior contains the origin, so outward normal has positive dot with
            # the centroid of the triangle
            centroid = (v1 + v2 + v3) / 3.0
            dot_centroid = np.dot(c_dir, centroid)
            if dot_centroid < 0:
                c = -c_dir
                dots = dots_neg
            else:
                c = c_dir
                dots = dots_pos
            
            # The circumradius is arccos(mean(dots))
            cos_r = np.mean(dots)
            cos_r = np.clip(cos_r, -1.0, 1.0)
            radii[i] = np.arccos(cos_r)
            centers[i] = c
        
        return radii, centers, simplices
    except Exception:
        return None, None, None

def _soft_max_logsumexp(x, beta=100.0):
    """
    Smooth approximation of max(x) using log-sum-exp.
    As beta -> infinity, this approaches max(x).
    """
    x_max = np.max(x)
    if x_max > 1e10:
        return x_max
    return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta

def _smooth_covering_radius(points, beta=100.0):
    """
    Smooth surrogate for the covering radius using log-sum-exp.
    """
    try:
        vor = SphericalVoronoi(points)
        vertices = vor.vertices
        dots = vertices @ points.T
        dots = np.clip(dots, -1.0, 1.0)
        min_dots = np.max(dots, axis=1)
        min_dists = np.arccos(min_dots)
        return _soft_max_logsumexp(min_dists, beta)
    except Exception:
        return np.pi / 2.0

def _smooth_grad_covering_radius(points, beta=100.0):
    """
    Compute gradient of the smooth covering radius surrogate.
    Uses finite differences or analytical gradient.
    """
    eps = 1e-6
    grad = np.zeros_like(points)
    
    try:
        vor = SphericalVoronoi(points)
        vertices = vor.vertices
        dots = vertices @ points.T
        dots = np.clip(dots, -1.0, 1.0)
        min_dots = np.max(dots, axis=1)
        min_dists = np.arccos(min_dots)
        
        # Soft max weights
        x_max = np.max(min_dists)
        weights = np.exp(beta * (min_dists - x_max))
        weights = weights / np.sum(weights)
        
        # For each Voronoi vertex, find which point is closest
        # and compute gradient contribution
        for i in range(N):
            # Find Voronoi vertices where point i is the nearest
            # This requires tracking which point is argmax for each Voronoi vertex
            pass
        
        # Use finite differences for simplicity
        for i in range(N):
            for j in range(3):
                orig = points[i, j]
                points[i, j] = orig + eps
                # Re-normalize
                norm = np.linalg.norm(points[i])
                points[i] = points[i] / norm
                f_plus = _smooth_covering_radius(points, beta)
                
                points[i, j] = orig - eps
                norm = np.linalg.norm(points[i])
                points[i] = points[i] / norm
                f_minus = _smooth_covering_radius(points, beta)
                
                points[i, j] = orig
                norm = np.linalg.norm(points[i])
                points[i] = points[i] / norm
                
                grad[i, j] = (f_plus - f_minus) / (2 * eps)
        
        # Project gradient onto tangent space
        for i in range(N):
            grad[i] = grad[i] - np.dot(grad[i], points[i]) * points[i]
        
        return grad
    except Exception:
        return grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    best_points = points.copy()
    best_radius = _compute_covering_radius(points)

    # 2. Lloyd's Algorithm (Spherical K-Means)
    for k in range(30):
        if time.time() > deadline - 40.0:
            break

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

            diff = np.linalg.norm(new_points - points, axis=1)
            if np.max(diff) < 1e-6:
                break

            points = new_points
            current_radius = _compute_covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()

        except Exception:
            break

    # 3. Balancing force: equalize facet circumradii
    # For each point, compute the average circumradius of adjacent facets.
    # If a facet has a large circumradius, its circumcenter is far from the points.
    # Moving a point toward the circumcenter of a large-circumradius facet
    # reduces that facet's circumradius.
    # The "force" on point i is the weighted sum of (center_f - point_i)
    # for all facets f containing point i, weighted by how much larger
    # facet f's circumradius is compared to the mean.
    step_size = 0.05
    for it in range(300):
        if time.time() > deadline - 30.0:
            break

        try:
            radii, centers, simplices = _compute_facet_circumradii_and_centers(points)
            if radii is None:
                break

            F = len(radii)
            mean_r = np.mean(radii)
            # Weight: how much above average each facet's circumradius is
            weights = np.maximum(radii - mean_r, 0.0)
            w_sum = np.sum(weights)
            if w_sum < 1e-15:
                break
            weights = weights / w_sum

            # Compute force on each point
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

            # Normalize forces and take a step
            force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
            force_norms[force_norms == 0] = 1.0
            unit_forces = forces / force_norms

            new_points = points + step_size * unit_forces
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            new_points = new_points / norms

            new_radius = _compute_covering_radius(new_points)

            if new_radius < best_radius:
                best_radius = new_radius
                best_points = new_points.copy()
                points = new_points
                step_size = min(step_size * 1.3, 0.15)
            else:
                step_size *= 0.6
                if step_size < 1e-5:
                    break

        except Exception:
            break

    # 4. Riemannian gradient flow with smooth surrogate
    if time.time() < deadline - 20.0:
        beta = 50.0
        lr = 0.01
        for it in range(100):
            if time.time() > deadline - 15.0:
                break
            
            try:
                grad = _smooth_grad_covering_radius(points, beta)
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
                    lr = min(lr * 1.1, 0.05)
                else:
                    lr *= 0.7
                    if lr < 1e-6:
                        break
                        
            except Exception:
                break

    # 5. Polish with Nelder-Mead on the covering radius
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
                       options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})

        if res.fun < best_radius:
            best_radius = res.fun
            best_flat = res.x
            norms = np.linalg.norm(best_flat.reshape(N, 3), axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            best_points = best_flat.reshape(N, 3) / norms
    except Exception:
        pass

    # Final check and return
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_points = best_points / norms

    return best_points
