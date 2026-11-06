"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

# Hard deadline: 145 seconds (5s safety margin for 150s kill)
DEADLINE = 145.0


def _covering_radius_rad(points):
    """
    Compute the covering radius (in radians) of points on the unit sphere.
    The covering radius is the maximum over all y on the sphere of min_i ang_dist(y, p_i).
    This is equivalent to the maximum circumradius of the spherical caps at the facets
    of the convex hull of the points.
    """
    # Normalize points
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    points = points / norms
    
    # Compute convex hull to find facets
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use all pairwise distances
        dots = np.clip(points @ points.T, -1.0, 1.0)
        dists = np.arccos(dots)
        return np.max(dists)
    
    # For each facet (triangle), compute the circumradius
    # The circumradius of a spherical triangle can be computed from its angles
    max_r = 0.0
    for simplex in hull.simplices:
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # Compute side lengths (angular distances)
        ab = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
        bc = np.arccos(np.clip(np.dot(b, c), -1.0, 1.0))
        ca = np.arccos(np.clip(np.dot(c, a), -1.0, 1.0))
        
        # Use spherical law of cosines to find circumradius
        # For a spherical triangle with sides a, b, c and circumradius R:
        # cos(a) = cos(b)cos(c) + sin(b)sin(c)cos(A)
        # We need to find the point equidistant from all three vertices
        
        # Alternative: find the circumcenter by solving for point equidistant from a, b, c
        # The circumcenter lies on the normal to the plane through a, b, c
        # and is equidistant from all three points
        
        # Compute normal to the plane
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            # Degenerate triangle
            r = max(ab, bc, ca) / 2
            max_r = max(max_r, r)
            continue
        n = n / n_norm
        
        # The circumcenter is of the form: a + t*n where t is chosen so that
        # |a + t*n - b| = |a + t*n - c| = |a + t*n|
        # This gives us equations to solve for t
        
        # Let's use a different approach: parameterize points on the great circle
        # perpendicular to the plane, and find the one equidistant from a, b, c
        
        # Actually, let's use the formula for circumradius of a spherical triangle
        # Given sides a, b, c (angular), the circumradius R satisfies:
        # cos(a) = cos(b)cos(c) + sin(b)sin(c)cos(A)
        # And by the spherical law of sines: sin(A)/sin(a) = sin(B)/sin(b) = sin(C)/sin(c) = 1/(2*sin(R))
        
        # Let's compute using the formula:
        # cos(2R) = (cos(a)cos(b)cos(c) - cos(a) - cos(b) - cos(c) + 1) / (2*(1 - cos(a) - cos(b) - cos(c) + cos(a)cos(b) + cos(b)cos(c) + cos(c)cos(a)))
        
        cos_a, cos_b, cos_c = np.cos(ab), np.cos(bc), np.cos(ca)
        
        numerator = cos_a * cos_b * cos_c - cos_a - cos_b - cos_c + 1
        denominator = 2 * (1 - cos_a - cos_b - cos_c + cos_a * cos_b + cos_b * cos_c + cos_c * cos_a)
        
        if abs(denominator) < 1e-10:
            # Degenerate case
            r = max(ab, bc, ca) / 2
            max_r = max(max_r, r)
            continue
        
        cos_2r = numerator / denominator
        cos_2r = np.clip(cos_2r, -1.0, 1.0)
        r = 0.5 * np.arccos(cos_2r)
        
        max_r = max(max_r, r)
    
    return max_r


def _objective_and_grad(x):
    """Objective function and gradient for optimization."""
    points = x.reshape(-1, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    points = points / norms
    
    # Compute covering radius
    r = _covering_radius_rad(points)
    
    # For gradient, we need to identify the critical facet
    # This is complex, so we'll use numerical gradient approximation
    # Or use a simpler objective: maximize minimum pairwise distance
    
    # Simpler objective: maximize the minimum angular distance between any two points
    dots = np.clip(points @ points.T, -1.0, 1.0)
    dists = np.arccos(dots)
    np.fill_diagonal(dists, np.pi)
    min_dist = np.min(dists)
    
    # Objective is negative of min_dist (we want to maximize it)
    obj = -min_dist
    
    # Gradient: approximate numerically
    grad = np.zeros_like(x)
    eps = 1e-7
    for i in range(x.size):
        x_plus = x.copy()
        x_plus[i] += eps
        obj_plus = _objective_and_grad(x_plus)[0]
        grad[i] = (obj_plus - obj) / eps
    
    return obj, grad


def construct():
    start_time = time.time()
    
    # Start from Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimize using L-BFGS-B
    best_points = points.copy()
    best_radius = _covering_radius_rad(best_points)
    
    # Try optimization
    try:
        def obj(x):
            pts = x.reshape(-1, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            pts = pts / norms
            return _covering_radius_rad(pts)
        
        result = minimize(
            obj,
            points.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 1000, 'maxfun': 5000}
        )
        
        if result.success or result.fun < best_radius:
            opt_points = result.x.reshape(-1, 3)
            opt_radius = _covering_radius_rad(opt_points)
            if opt_radius < best_radius:
                best_points = opt_points
                best_radius = opt_radius
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 10:
        return best_points
    
    # Try additional local optimization with multiple restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Perturb the best solution
        noise = np.random.randn(N, 3) * 0.1
        perturbed = best_points + noise
        # Renormalize
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed = perturbed / norms
        
        try:
            result = minimize(
                obj,
                perturbed.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 2000}
            )
            
            if result.fun < best_radius:
                opt_points = result.x.reshape(-1, 3)
                opt_radius = _covering_radius_rad(opt_points)
                if opt_radius < best_radius:
                    best_points = opt_points
                    best_radius = opt_radius
        except Exception:
            pass
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
