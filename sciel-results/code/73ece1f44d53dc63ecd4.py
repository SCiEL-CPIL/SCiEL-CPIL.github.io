import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 130

def _fibonacci_spiral(n):
    """Generate n points on the unit sphere using a Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _normalize(v):
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms

def _covering_radius_grid(points, grid_points, beta=50.0):
    """
    Compute a smooth approximation of the covering radius using a grid of points.
    We minimize the max angular distance. 
    Angular distance = arccos(dot).
    We use soft-max of arccos(dot) values.
    """
    # points: (N, 3)
    # grid_points: (M, 3)
    # Compute dot products: (M, N)
    dots = grid_points @ points.T
    # Clip to avoid arccos domain errors
    dots = np.clip(dots, -1.0, 1.0)
    # Angular distances: (M, N)
    dists = np.arccos(dots)
    # For each grid point, find the min distance to any of the N points
    min_dists = np.min(dists, axis=1)
    # Soft-max of min_dists to approximate the max covering radius
    # max(x) ≈ (1/beta) * log(sum(exp(beta * x)))
    max_dist = (1.0 / beta) * np.log(np.sum(np.exp(beta * min_dists)))
    return max_dist

def _grad_covering_radius_grid(points, grid_points, beta=50.0):
    """
    Gradient of the smooth covering radius approximation.
    """
    dots = grid_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    
    # Soft-max weights
    exp_dists = np.exp(beta * min_dists)
    sum_exp = np.sum(exp_dists)
    weights = exp_dists / sum_exp # (M,)
    
    # Gradient of min_dists w.r.t. points
    # d/dp arccos(g.p) = -g / sqrt(1 - (g.p)^2)
    # We need the gradient of the min over k. The min is achieved at the nearest point.
    # Let k_i be the index of the nearest point for grid point i.
    min_indices = np.argmin(dists, axis=1)
    
    grad = np.zeros_like(points)
    for i in range(len(grid_points)):
        k = min_indices[i]
        g = grid_points[i]
        p = points[k]
        dot = np.dot(g, p)
        dot = np.clip(dot, -1.0, 1.0)
        denom = np.sqrt(1.0 - dot*dot + 1e-12)
        # d dist / d p = -g / denom
        grad[k] += weights[i] * beta * (-g / denom)
        
    return grad

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin
    
    # 1. Initial Configuration
    pts = _fibonacci_spiral(N)
    
    # 2. Dense Grid for Surrogate
    # Use a dense Fibonacci spiral for the grid
    M = 10000
    grid_pts = _fibonacci_spiral(M)
    
    # 3. Optimization using L-BFGS-B
    # We flatten the points for the optimizer
    x0 = pts.flatten()
    
    def objective(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        return _covering_radius_grid(p, grid_pts, beta=50.0)
        
    def gradient(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        g = _grad_covering_radius_grid(p, grid_pts, beta=50.0)
        return g.flatten()
        
    try:
        res = minimize(objective, x0, method='L-BFGS-B', jac=gradient, 
                       options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10})
        pts = _normalize(res.x.reshape(N, 3))
    except Exception as e:
        # Fallback to original if optimization fails
        pts = _fibonacci_spiral(N)

    # 4. Geometric Polish using Convex Hull
    # This step corrects for any "holes" not well-captured by the grid and refines the worst facets.
    max_iter = 50
    for _ in range(max_iter):
        if time.time() - start_time > deadline - 5.0:
            break
        try:
            hull = ConvexHull(pts)
            facets = hull.simplices
            worst_idx = -1
            worst_radius = 0.0
            worst_circumcenter = None
            worst_facet_pts = None
            
            for fac in facets:
                p1, p2, p3 = pts[fac]
                # Compute circumcenter of triangle on sphere
                # The circumcenter is along the normal to the plane of the triangle
                # Normal = cross(p2-p1, p3-p1)
                n = np.cross(p2 - p1, p3 - p1)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-10:
                    continue
                n = n / n_norm
                
                # The circumcenter y is a scalar multiple of n.
                # We need y such that |y - p1| = |y - p2| = |y - p3|.
                # y = k * n.
                # |k*n - p1|^2 = k^2 - 2k(n.p1) + |p1|^2 = k^2 - 2k(n.p1) + 1
                # This must be equal for p1, p2, p3.
                # So n.p1 = n.p2 = n.p3 = c.
                # k is determined by the radius.
                # Actually, for a spherical cap, the "circumcenter" in the context of covering radius
                # is the point on the sphere farthest from the facet vertices.
                # It is the intersection of the sphere with the normal vector from the centroid?
                # No, it's the point y on the sphere that maximizes min(|y-pi|).
                # For a facet, this is the circumcenter of the spherical triangle.
                
                # Let's compute the circumcenter of the spherical triangle defined by p1, p2, p3.
                # It lies on the normal n.
                # Let y = n. (normalized)
                # The angular distance from y to p1 is arccos(n.p1).
                # This is the covering radius for this facet.
                
                dist = np.arccos(np.clip(np.dot(n, p1), -1.0, 1.0))
                if dist > worst_radius:
                    worst_radius = dist
                    worst_idx = len(facets)
                    worst_circumcenter = n
                    worst_facet_pts = pts[fac]
                    worst_facet_indices = fac

            if worst_idx == -1:
                break
                
            # Move the worst facet points slightly towards the circumcenter
            # to reduce the circumradius.
            # Gradient of angular distance w.r.t p_i is -y / sqrt(1 - (y.p_i)^2)
            # We want to minimize the max distance, so we move p_i in the direction of y.
            
            alpha = 0.05 # Step size
            for idx in worst_facet_indices:
                p = pts[idx]
                y = worst_circumcenter
                dot = np.dot(p, y)
                dot = np.clip(dot, -1.0, 1.0)
                denom = np.sqrt(1.0 - dot*dot + 1e-12)
                # Direction to move p to increase dot(p,y) is y - p (tangent component)
                # Actually, just move towards y
                new_p = p + alpha * (y - p)
                pts[idx] = _normalize(new_p)
                
        except Exception:
            break
            
    # Final check: Ensure no coincident points and valid shape
    pts = _normalize(pts)
    
    # Check for coincident points (safety)
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-6:
                # Perturb
                pts[j] += np.random.normal(0, 0.01, 3)
                pts[j] = _normalize(pts[j])
                
    return pts
