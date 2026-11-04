"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """Compute the covering radius in degrees using convex hull facets.
    The covering radius is the maximum over all facets of the angular distance
    from the facet's circumcenter (on the sphere) to the farthest vertex.
    """
    from scipy.spatial import ConvexHull
    
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
        # For each triangular facet, compute the circumcenter on the sphere
        # and the angular distance to each vertex
        max_circumradius = 0.0
        
        for simplex in hull.simplices:
            # Get the three points forming this triangle
            tri = pts[simplex]  # shape (3, 3)
            
            # Compute the circumcenter of the spherical triangle
            # The circumcenter is the point on the sphere equidistant (angularly) from all three vertices
            # This is the normalized cross product of (p2-p1) x (p3-p1) projected to sphere
            # Actually, for a spherical triangle, the circumcenter is the intersection of the sphere
            # with the plane through the three points... no, that's the planar circumcenter.
            
            # Better approach: the circumcenter of a spherical triangle is the point c on the unit sphere
            # such that c·p1 = c·p2 = c·p3 (equal dot products = equal angular distances)
            # This means c is in the null space of the matrix with rows [p1-p2, p2-p3, p3-p1]
            # Or equivalently, c is parallel to (p1 x p2) + (p2 x p3) + (p3 x p1) for equilateral
            # More generally, solve: c·p1 = c·p2 = c·p3, |c|=1
            
            # Set up: c·p1 = c·p2 => c·(p1-p2) = 0
            # c·p2 = c·p3 => c·(p2-p3) = 0
            # So c is in the intersection of two planes, i.e., c is parallel to (p1-p2) x (p2-p3)
            
            v1 = tri[0] - tri[1]
            v2 = tri[1] - tri[2]
            c = np.cross(v1, v2)
            
            if np.linalg.norm(c) < 1e-10:
                # Degenerate case, use centroid
                c = np.mean(tri, axis=0)
            
            c = c / np.linalg.norm(c)
            
            # Compute angular distances from c to each vertex
            dots = np.dot(tri, c)
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots)
            circumradius = np.max(dists)
            
            if circumradius > max_circumradius:
                max_circumradius = circumradius
        
        return np.degrees(max_circumradius)
    except Exception:
        # Fallback: dense sampling
        n_test = 20000
        i_test = np.arange(n_test) + 0.5
        z_test = 1.0 - 2.0 * i_test / n_test
        phi_test = np.pi * (1.0 + np.sqrt(5.0)) * i_test
        r_test = np.sqrt(1.0 - z_test * z_test)
        test_pts = np.column_stack([r_test * np.cos(phi_test), r_test * np.sin(phi_test), z_test])
        
        max_dist = 0.0
        batch_size = 5000
        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            batch = test_pts[start:end]
            dots = batch @ pts.T
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots)
            min_dists = np.min(dists, axis=1)
            batch_max = np.max(min_dists)
            if batch_max > max_dist:
                max_dist = batch_max
        return np.degrees(max_dist)


def _smooth_objective(flat_pts, n_test=5000):
    """Smooth objective: variance of nearest-neighbor distances + soft max.
    This encourages equiangularity and penalizes large covering radius.
    """
    pts = _normalize(flat_pts.reshape(N, 3))
    
    # Generate test points (Fibonacci spiral)
    i_test = np.arange(n_test) + 0.5
    z_test = 1.0 - 2.0 * i_test / n_test
    phi_test = np.pi * (1.0 + np.sqrt(5.0)) * i_test
    r_test = np.sqrt(1.0 - z_test * z_test)
    test_pts = np.column_stack([r_test * np.cos(phi_test), r_test * np.sin(phi_test), z_test])
    
    # Compute angular distances
    dots = test_pts @ pts.T  # (n_test, N)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)  # (n_test, N)
    
    # For each test point, find nearest point and its distance
    min_dists = np.min(dists, axis=1)
    
    # Variance of distances (encourages equiangularity)
    mean_dist = np.mean(min_dists)
    var = np.mean((min_dists - mean_dist) ** 2)
    
    # Soft max to penalize large covering radius
    beta = 40.0
    log_sum_exp = np.log(np.sum(np.exp(beta * min_dists))) / beta
    
    # Combined objective
    obj = var + 0.15 * log_sum_exp
    
    return obj


def _gradient(flat_pts, n_test=5000, eps=1e-7):
    """Compute gradient using central finite differences (expensive but accurate)."""
    grad = np.zeros_like(flat_pts)
    for i in range(len(flat_pts)):
        fp = flat_pts.copy()
        fm = flat_pts.copy()
        fp[i] += eps
        fm[i] -= eps
        grad[i] = (_smooth_objective(fp, n_test) - _smooth_objective(fm, n_test)) / (2 * eps)
    return grad


def _voronoi_centroid_refine(pts, n_test=15000, n_iter=50, step=0.05):
    """Refine points by moving them toward their Voronoi cell centroids."""
    pts = _normalize(pts.copy())
    
    # Generate test points
    i_test = np.arange(n_test) + 0.5
    z_test = 1.0 - 2.0 * i_test / n_test
    phi_test = np.pi * (1.0 + np.sqrt(5.0)) * i_test
    r_test = np.sqrt(1.0 - z_test * z_test)
    test_pts = np.column_stack([r_test * np.cos(phi_test), r_test * np.sin(phi_test), z_test])
    
    for iteration in range(n_iter):
        # Compute nearest neighbor for each test point
        dots = test_pts @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        nearest = np.argmin(dists, axis=1)
        
        # Compute Voronoi cell centroids using bincount for efficiency
        # For each dimension, sum test_pts where nearest == j, then divide by count
        new_pts = np.zeros_like(pts)
        for dim in range(3):
            sums = np.bincount(nearest, weights=test_pts[:, dim], minlength=N)
            counts = np.bincount(nearest, minlength=N)
            counts[counts == 0] = 1  # avoid division by zero
            new_pts[:, dim] = sums / counts
        
        # Normalize centroids
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        new_pts = new_pts / norms
        
        # Blend with current points
        pts = (1 - step) * pts + step * new_pts
        pts = _normalize(pts)
        
        # Check convergence
        if iteration > 0:
            change = np.linalg.norm(pts - _normalize(new_pts))
            if change < 1e-10:
                break
    
    return pts


def construct():
    """Construct 130 points on the unit sphere minimizing covering radius."""
    best_pts = None
    best_radius = np.inf
    
    # Try multiple initializations
    n_restarts = 3
    
    for restart in range(n_restarts):
        if restart == 0:
            # Fibonacci spiral
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            pts0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        elif restart == 1:
            # Random initialization with some structure
            rng = np.random.default_rng(42 + restart)
            # Start with Fibonacci and add small random perturbations
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.normal(0, 0.1, N)
            r = np.sqrt(1.0 - z * z)
            pts0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            pts0 = _normalize(pts0)
        else:
            # More aggressive random perturbation
            rng = np.random.default_rng(42 + restart)
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.normal(0, 0.3, N)
            r = np.sqrt(1.0 - z * z)
            pts0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            pts0 = _normalize(pts0)
        
        # First: Voronoi centroid refinement (fast, good for local optimization)
        pts_refined = _voronoi_centroid_refine(pts0, n_test=10000, n_iter=30, step=0.1)
        
        # Second: Smooth optimization using L-BFGS-B
        # Use a smaller number of test points for speed during optimization
        def _obj_and_grad(flat):
            obj = _smooth_objective(flat, n_test=4000)
            # Use numerical gradient with smaller perturbation
            grad = np.zeros_like(flat)
            eps = 1e-6
            for i in range(0, len(flat), 3):  # Optimize in groups of 3 for efficiency
                for j in range(i, min(i+3, len(flat))):
                    fp = flat.copy()
                    fm = flat.copy()
                    fp[j] += eps
                    fm[j] -= eps
                    grad[j] = (_smooth_objective(fp, n_test=2000) - _smooth_objective(fm, n_test=2000)) / (2 * eps)
            return obj, grad
        
        flat0 = pts_refined.flatten()
        
        try:
            res = minimize(_obj_and_grad, flat0, method='L-BFGS-B',
                         jac=True, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            pts_opt = _normalize(res.x.reshape(N, 3))
        except Exception:
            pts_opt = pts_refined
        
        # Final refinement with more Voronoi iterations
        pts_final = _voronoi_centroid_refine(pts_opt, n_test=20000, n_iter=100, step=0.03)
        
        # Compute covering radius
        radius = _covering_radius(pts_final)
        
        if radius < best_radius:
            best_radius = radius
            best_pts = pts_final
    
    return best_pts
# EVOLVE-BLOCK-END
