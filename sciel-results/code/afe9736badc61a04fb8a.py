"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _covering_radius(points):
    """Compute the covering radius in radians for points on the unit sphere."""
    points = np.asarray(points)
    # Remove any non-finite
    if not np.all(np.isfinite(points)):
        return np.inf
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    # Check for coincident points
    from scipy.spatial.distance import pdist
    d = pdist(points)
    if np.any(d < 1e-10):
        # Allow very close but not identical
        pass
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.inf
    
    # For each facet, compute the circumcenter and check if it's on the sphere
    # The covering radius is the max over facets of the angular distance from circumcenter to facet vertices
    # For a spherical convex hull, the covering radius is determined by the largest empty cap
    # which corresponds to the circumcircle of the largest facet
    
    max_ang = 0.0
    for simplex in hull.simplices:
        facet_pts = points[simplex]
        # Compute circumcenter of the facet on the sphere
        # For a triangle on the unit sphere, the circumcenter is the normalized cross product of (v1-v3) x (v2-v3)
        # Actually, the circumcenter of a spherical triangle is in the direction of the normalized 
        # (v1 x v2 + v2 x v3 + v3 x v1) or we can solve the linear system
        
        # For a triangle with vertices a, b, c on the unit sphere, the circumcenter o satisfies:
        # o . a = o . b = o . c = cos(alpha)
        # This means o is in the span of a, b, c and o . (a-b) = 0, o . (b-c) = 0
        # So o is parallel to (a-b) x (b-c)
        
        a, b, c = facet_pts
        v1 = a - b
        v2 = b - c
        o = np.cross(v1, v2)
        o_norm = np.linalg.norm(o)
        if o_norm < 1e-15:
            continue
        o = o / o_norm
        
        # Check which side of the facet the origin is on
        # The circumcenter should be on the opposite side from the origin for the cap to be "empty"
        # Actually, for the covering radius, we want the circumcenter that is farthest from the facet vertices
        # There are two circumcenters (o and -o); we pick the one that is on the "outside" of the hull
        
        # The normal to the facet pointing outward from the convex hull
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-15:
            continue
        n = n / n_norm
        
        # The outward normal should point away from the origin
        # For a convex hull containing the origin, the outward normal has negative dot with origin... 
        # Actually, for points on the unit sphere, the convex hull contains the origin if and only if
        # the points are not all in a hemisphere. The outward normal of a facet points away from the interior.
        # The circumcenter of the "empty cap" is the one on the same side as the outward normal.
        
        # Let's check: if o . n > 0, then o is on the outward side
        if o.dot(n) < 0:
            o = -o
        
        # Now compute the angular distance from o to the facet vertices
        # This should be the radius of the empty cap
        for v in facet_pts:
            cos_ang = np.clip(o.dot(v), -1.0, 1.0)
            ang = np.arccos(cos_ang)
            if ang > max_ang:
                max_ang = ang
    
    return max_ang


def _covering_radius_grad(points):
    """Compute covering radius and gradient with respect to points."""
    points = np.asarray(points)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.inf, None
    
    max_ang = 0.0
    max_facet = None
    max_vertex = None
    
    for idx, simplex in enumerate(hull.simplices):
        facet_pts = points[simplex]
        a, b, c = facet_pts
        v1 = a - b
        v2 = b - c
        o = np.cross(v1, v2)
        o_norm = np.linalg.norm(o)
        if o_norm < 1e-15:
            continue
        o = o / o_norm
        
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-15:
            continue
        n = n / n_norm
        
        if o.dot(n) < 0:
            o = -o
        
        for vi, v in enumerate(facet_pts):
            cos_ang = np.clip(o.dot(v), -1.0, 1.0)
            ang = np.arccos(cos_ang)
            if ang > max_ang:
                max_ang = ang
                max_facet = simplex
                max_vertex = vi
    
    return max_ang, (max_facet, max_vertex, points.copy())


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    # Local optimization using gradient descent on the sphere
    # We use the covering radius as the objective and minimize it
    
    def objective_and_grad(x):
        points = x.reshape(N, 3)
        # Project to sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        points = points / norms
        
        ang, info = _covering_radius_grad(points)
        if ang == np.inf or info is None:
            return ang, np.zeros_like(x)
        
        max_facet, max_vertex, pts = info
        facet_pts = pts[max_facet]
        a, b, c = facet_pts
        
        # Compute circumcenter
        v1 = a - b
        v2 = b - c
        o = np.cross(v1, v2)
        o_norm = np.linalg.norm(o)
        if o_norm < 1e-15:
            return ang, np.zeros_like(x)
        o = o / o_norm
        
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-15:
            return ang, np.zeros_like(x)
        n = n / n_norm
        
        if o.dot(n) < 0:
            o = -o
        
        # The gradient of the covering radius with respect to the points
        # is related to moving the circumcenter toward the facet center
        # For now, use a simple heuristic: move points toward the circumcenter
        # to reduce the angular distance
        
        grad = np.zeros_like(x)
        # Move each point in the facet toward the circumcenter
        for vi, idx in enumerate(max_facet):
            v = pts[idx]
            # Gradient direction: move v toward o
            # The angular distance is arccos(o . v), so d/dv = -o / sqrt(1 - (o.v)^2)
            cos_val = np.clip(o.dot(v), -1.0, 1.0)
            sin_val = np.sqrt(max(0, 1.0 - cos_val * cos_val))
            if sin_val > 1e-10:
                grad[idx] = -o / sin_val
        
        return ang, grad
    
    # Use L-BFGS-B with the sphere constraint handled by projection
    best_radius = _covering_radius(best_points)
    
    # Iterative local search
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        # Try to improve using gradient descent
        x = best_points.flatten()
        
        def obj(x_flat):
            p = x_flat.reshape(N, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            p = p / norms
            return _covering_radius(p)
        
        result = minimize(
            obj,
            x,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 1000},
            tol=1e-8
        )
        
        new_points = result.x.reshape(N, 3)
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        new_points = new_points / norms
        
        new_radius = _covering_radius(new_points)
        
        if new_radius < best_radius - 1e-10:
            best_points = new_points
            best_radius = new_radius
            print(f"Iteration {iteration}: radius = {np.degrees(best_radius):.6f} deg")
        
        # Add small perturbation to escape local minima
        if time.time() > deadline:
            break
        
        noise = 0.01 * np.random.randn(N, 3)
        perturbed = best_points + noise
        perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
        
        # Try the perturbed version
        pert_radius = _covering_radius(perturbed)
        if pert_radius < best_radius - 1e-10:
            best_points = perturbed
            best_radius = pert_radius
    
    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
