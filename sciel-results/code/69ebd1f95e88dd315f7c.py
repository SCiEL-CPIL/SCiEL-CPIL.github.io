import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130


def _covering_radius(points):
    """
    Compute the covering radius in degrees using convex hull facets.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        p1 = points[facets[:, 0]]
        p2 = points[facets[:, 1]]
        p3 = points[facets[:, 2]]
        
        v1 = p1 - p2
        v2 = p1 - p3
        
        c_dir = np.cross(v1, v2)
        norms = np.linalg.norm(c_dir, axis=1)
        
        valid = norms > 1e-10
        if not np.any(valid):
            return 90.0
            
        c = np.zeros_like(c_dir)
        c[valid] = c_dir[valid] / norms[valid, np.newaxis]
        
        centroids = (p1 + p2 + p3) / 3.0
        c_norms = np.linalg.norm(centroids, axis=1)
        valid_centroids = c_norms > 1e-10
        centroids[valid_centroids] /= c_norms[valid_centroids, np.newaxis]
        
        dots = np.sum(c * centroids, axis=1)
        c[dots < 0] *= -1
        
        cos_theta = np.sum(c * p1, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        theta[~valid] = 0.0
        return float(np.max(theta) * 180.0 / np.pi)
        
    except Exception:
        return 180.0


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points


def _lloyd_step(points, step_size=0.1):
    """
    One step of Lloyd's algorithm for spherical Voronoi:
    Move each point toward the centroid of its Voronoi cell (approximated by
    the centroid of its Delaunay facets).
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        # For each point, accumulate the centroid of its adjacent facets
        new_points = np.zeros_like(points)
        counts = np.zeros(N)
        
        for facet in facets:
            centroid = (points[facet[0]] + points[facet[1]] + points[facet[2]]) / 3.0
            for idx in facet:
                new_points[idx] += centroid
                counts[idx] += 1
        
        # Avoid division by zero
        valid = counts > 0
        new_points[valid] /= counts[valid, np.newaxis]
        
        # Move points toward their new positions
        points_new = points.copy()
        for i in range(N):
            if valid[i]:
                d = new_points[i] - points[i]
                norm_d = np.linalg.norm(d)
                if norm_d > 1e-12:
                    points_new[i] = points[i] + step_size * d / norm_d
        
        # Normalize
        norms = np.linalg.norm(points_new, axis=1, keepdims=True)
        valid_norms = norms > 1e-10
        points_new[valid_norms] /= norms[valid_norms]
        
        return points_new
    except Exception:
        return points.copy()


def _softmax_gradient_step(points, tau=50.0, step_size=0.03):
    """
    One step of gradient descent on the soft-max of circumradii.
    Uses finite-difference gradient approximation for robustness.
    """
    def objective(pts):
        try:
            hull = ConvexHull(pts)
            facets = hull.simplices
            p1 = pts[facets[:, 0]]
            p2 = pts[facets[:, 1]]
            p3 = pts[facets[:, 2]]
            
            v1 = p1 - p2
            v2 = p1 - p3
            
            c_dir = np.cross(v1, v2)
            norms_c = np.linalg.norm(c_dir, axis=1)
            
            valid = norms_c > 1e-10
            if not np.any(valid):
                return 180.0
                
            c = np.zeros_like(c_dir)
            c[valid] = c_dir[valid] / norms_c[valid, np.newaxis]
            
            centroids = (p1 + p2 + p3) / 3.0
            c_norms = np.linalg.norm(centroids, axis=1)
            valid_centroids = c_norms > 1e-10
            centroids[valid_centroids] /= c_norms[valid_centroids, np.newaxis]
            
            dots = np.sum(c * centroids, axis=1)
            c[dots < 0] *= -1
            
            cos_theta = np.sum(c * p1, axis=1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            
            theta[~valid] = 0.0
            
            max_theta = np.max(theta)
            exp_terms = np.exp(tau * (theta - max_theta))
            soft_max = (1.0 / tau) * (max_theta + np.log(np.sum(exp_terms)))
            
            return float(soft_max * 180.0 / np.pi)
        except Exception:
            return 180.0
    
    # Compute gradient via finite differences
    eps = 1e-5
    grad = np.zeros_like(points)
    x0 = points.flatten()
    
    for i in range(len(x0)):
        x_plus = x0.copy()
        x_plus[i] += eps
        pts_plus = x_plus.reshape(N, 3)
        norms = np.linalg.norm(pts_plus, axis=1, keepdims=True)
        valid = norms > 1e-10
        pts_plus[valid] /= norms[valid]
        
        x_minus = x0.copy()
        x_minus[i] -= eps
        pts_minus = x_minus.reshape(N, 3)
        norms = np.linalg.norm(pts_minus, axis=1, keepdims=True)
        valid = norms > 1e-10
        pts_minus[valid] /= norms[valid]
        
        f_plus = objective(pts_plus)
        f_minus = objective(pts_minus)
        grad_flat = np.zeros_like(x0)
        grad_flat[i] = (f_plus - f_minus) / (2 * eps)
        grad = grad_flat.reshape(N, 3)
        break  # This is too slow, use a simpler approach
    
    # Use a simpler heuristic: move points away from the worst facet
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        p1 = points[facets[:, 0]]
        p2 = points[facets[:, 1]]
        p3 = points[facets[:, 2]]
        
        v1 = p1 - p2
        v2 = p1 - p3
        
        c_dir = np.cross(v1, v2)
        norms_c = np.linalg.norm(c_dir, axis=1)
        
        valid = norms_c > 1e-10
        if not np.any(valid):
            return points.copy()
            
        c = np.zeros_like(c_dir)
        c[valid] = c_dir[valid] / norms_c[valid, np.newaxis]
        
        centroids = (p1 + p2 + p3) / 3.0
        c_norms = np.linalg.norm(centroids, axis=1)
        valid_centroids = c_norms > 1e-10
        centroids[valid_centroids] /= c_norms[valid_centroids, np.newaxis]
        
        dots = np.sum(c * centroids, axis=1)
        c[dots < 0] *= -1
        
        cos_theta = np.sum(c * p1, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        theta[~valid] = 0.0
        
        max_idx = np.argmax(theta)
        if not valid[max_idx]:
            return points.copy()
            
        f_idx = facets[max_idx]
        cc = c[max_idx]
        
        points_new = points.copy()
        for v in f_idx:
            d = cc - points[v]
            n = np.linalg.norm(d)
            if n > 1e-10:
                points_new[v] += step_size * (d / n)
        
        norms = np.linalg.norm(points_new, axis=1, keepdims=True)
        valid_norms = norms > 1e-10
        points_new[valid_norms] /= norms[valid_norms]
        
        return points_new
    except Exception:
        return points.copy()


def _worst_facet_repair(points, max_iter=150, deadline=None):
    """
    Iteratively repair the worst facet by moving its vertices toward the
    facet's circumcenter.
    """
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        try:
            hull = ConvexHull(points)
            facets = hull.simplices
            p1 = points[facets[:, 0]]
            p2 = points[facets[:, 1]]
            p3 = points[facets[:, 2]]
            
            v1 = p1 - p2
            v2 = p1 - p3
            
            c_dir = np.cross(v1, v2)
            norms_c = np.linalg.norm(c_dir, axis=1)
            
            valid = norms_c > 1e-10
            if not np.any(valid):
                break
                
            c = np.zeros_like(c_dir)
            c[valid] = c_dir[valid] / norms_c[valid, np.newaxis]
            
            centroids = (p1 + p2 + p3) / 3.0
            c_norms = np.linalg.norm(centroids, axis=1)
            valid_centroids = c_norms > 1e-10
            centroids[valid_centroids] /= c_norms[valid_centroids, np.newaxis]
            
            dots = np.sum(c * centroids, axis=1)
            c[dots < 0] *= -1
            
            cos_theta = np.sum(c * p1, axis=1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
            theta[~valid] = 0.0
            
            max_idx = np.argmax(theta)
            if not valid[max_idx]:
                break
                
            f_idx = facets[max_idx]
            cc = c[max_idx]
            
            points_new = points.copy()
            for v in f_idx:
                d = cc - points[v]
                n = np.linalg.norm(d)
                if n > 1e-10:
                    points_new[v] += 0.01 * (d / n)
            
            norms = np.linalg.norm(points_new, axis=1, keepdims=True)
            valid_norms = norms > 1e-10
            points_new[valid_norms] /= norms[valid_norms]
            
            current_radius = _covering_radius(points_new)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points_new.copy()
                points = points_new
            else:
                # Try a smaller step
                points_new = points.copy()
                for v in f_idx:
                    d = cc - points[v]
                    n = np.linalg.norm(d)
                    if n > 1e-10:
                        points_new[v] += 0.005 * (d / n)
                
                norms = np.linalg.norm(points_new, axis=1, keepdims=True)
                valid_norms = norms > 1e-10
                points_new[valid_norms] /= norms[valid_norms]
                
                current_radius = _covering_radius(points_new)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = points_new.copy()
                    points = points_new
                else:
                    break
        except Exception:
            break
    
    return best_points, best_radius


def _random_sphere(n, seed=None):
    """Generate n random points on the unit sphere."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    points = rng.normal(size=(n, 3))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points /= norms
    return points


def _perturb_points(points, scale=0.1, seed=None):
    """Perturb points by adding Gaussian noise and re-normalizing."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    noise = rng.normal(scale=scale, size=points.shape)
    points_new = points + noise
    norms = np.linalg.norm(points_new, axis=1, keepdims=True)
    valid = norms > 1e-10
    points_new[valid] /= norms[valid]
    return points_new


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_points = None
    best_radius = 180.0
    
    # Multiple restarts with different initializations
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_fibonacci_sphere(N))
    
    # 2. Random initializations with different seeds
    for seed in [42, 123, 456, 789, 101112]:
        initializations.append(_random_sphere(N, seed=seed))
    
    # 3. Perturbed Fibonacci
    for scale in [0.05, 0.1, 0.2]:
        initializations.append(_perturb_points(_fibonacci_sphere(N), scale=scale, seed=scale * 100))
    
    for init_idx, init_points in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        
        points = init_points.copy()
        
        # Phase 0: Initial Lloyd's iterations for coarse optimization
        lloyd_iters = 60
        for it in range(lloyd_iters):
            if time.time() > deadline - 15.0:
                break
            points = _lloyd_step(points, step_size=0.1)
            current_radius = _covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
        
        # Phase 1: Soft-max gradient polishing (worst facet repair)
        if time.time() < deadline - 10.0:
            for it in range(80):
                if time.time() > deadline - 8.0:
                    break
                points = _softmax_gradient_step(points, tau=50.0, step_size=0.03)
                current_radius = _covering_radius(points)
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = points.copy()
        
        # Phase 2: Fine worst-facet repair
        if time.time() < deadline - 5.0:
            points_refined, local_radius = _worst_facet_repair(
                best_points.copy(), max_iter=200, deadline=deadline - 2.0
            )
            if local_radius < best_radius:
                best_radius = local_radius
                best_points = points_refined.copy()
    
    # Final safety check
    if best_points is None:
        best_points = _fibonacci_sphere(N)
    
    # Ensure all points are on the unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    valid = norms > 1e-10
    best_points[valid] /= norms[valid]
    
    return best_points


if __name__ == "__main__":
    pts = construct()
    print(f"Covering radius: {_covering_radius(pts)} degrees")
    print(pts.shape)
