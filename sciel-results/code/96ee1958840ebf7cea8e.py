"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _covering_radius(points):
    """Compute the covering radius in radians for a set of points on the unit sphere.
    
    The covering radius is the maximum over all points y on the sphere of the
    angular distance from y to the nearest point in the configuration.
    This is equivalent to the maximum circumradius of the facets of the convex hull
    (Voronoi cells on the sphere).
    
    We approximate this by computing the spherical Voronoi diagram via the convex hull
    of the points, then finding the circumradius of each facet.
    """
    from scipy.spatial import ConvexHull
    
    # Ensure points are on the unit sphere
    norms = np.linalg.norm(points, axis=1)
    points = points / norms
    
    # Compute the convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use a dense sampling approach
        n_samples = 5000
        # Generate random points on the sphere
        rng = np.random.default_rng(42)
        # Use Fibonacci spiral for better coverage
        i = np.arange(n_samples) + 0.5
        z = 1.0 - 2.0 * i / n_samples
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        sample_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # For each sample point, find the nearest configuration point
        # Using matrix multiplication for efficiency
        # cos(angle) = dot product for unit vectors
        dots = sample_points @ points.T
        # Clip to [-1, 1] for numerical safety
        dots = np.clip(dots, -1.0, 1.0)
        max_dots = dots.max(axis=1)
        min_angles = np.arccos(max_dots)
        return float(np.max(min_angles))
    
    # Get the facets (simplices) of the convex hull
    # Each facet is a set of indices into points
    max_radius = 0.0
    for simplex in hull.simplices:
        facet_points = points[simplex]
        # Compute the circumcenter of the facet
        # For a triangle (2-simplex) in 3D, the circumcenter lies on the plane of the triangle
        # We need the point on the sphere that is equidistant from all facet vertices
        # and lies in the direction of the facet's outward normal
        
        # The circumradius of a spherical triangle with vertices v1, v2, v3
        # is the angular distance from the circumcenter to any vertex.
        # The circumcenter is in the direction of the cross product based formula.
        
        # For a triangle with vertices a, b, c on the unit sphere:
        # The circumcenter direction is proportional to:
        # (b-a) x (c-a) ... no, that's the normal.
        # Actually, the circumcenter of a spherical triangle can be found as follows:
        # It is the point x on the sphere such that x·a = x·b = x·c.
        # This means x is in the null space of [a-b; b-c] (in the sense of dot products).
        # More precisely, x is orthogonal to (a-b) and (b-c) in the Euclidean sense? No.
        # x·a = x·b => x·(a-b) = 0
        # x·b = x·c => x·(b-c) = 0
        # So x is orthogonal to (a-b) and (b-c) in 3D Euclidean space.
        # Thus x is parallel to (a-b) x (b-c) = (a-b) x (c-b)
        
        if len(simplex) == 3:
            a, b, c = facet_points
            # x is parallel to (a-b) x (c-b)
            v1 = a - b
            v2 = c - b
            x = np.cross(v1, v2)
            norm_x = np.linalg.norm(x)
            if norm_x < 1e-12:
                # Degenerate triangle
                continue
            x = x / norm_x
            
            # Check which hemisphere: we want the circumcenter that is on the same side
            # as the facet normal (outward from the convex hull)
            # The facet normal points outward from the convex hull
            facet_normal = np.cross(b - a, c - a)
            facet_normal_norm = np.linalg.norm(facet_normal)
            if facet_normal_norm < 1e-12:
                continue
            facet_normal = facet_normal / facet_normal_norm
            
            # The circumcenter should be in the direction of the facet normal
            # (for the covering, we care about the "empty cap" side)
            if np.dot(x, facet_normal) < 0:
                x = -x
            
            # The circumradius is the angular distance from x to any vertex
            # Use the average for numerical stability
            dots = np.abs(np.dot(x, a))
            dots = np.clip(dots, -1.0, 1.0)
            radius = np.arccos(dots)
            max_radius = max(max_radius, radius)
    
    if max_radius == 0.0:
        # Fallback if no valid facets
        n_samples = 5000
        i = np.arange(n_samples) + 0.5
        z = 1.0 - 2.0 * i / n_samples
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        sample_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        dots = sample_points @ points.T
        dots = np.clip(dots, -1.0, 1.0)
        max_dots = dots.max(axis=1)
        min_angles = np.arccos(max_dots)
        max_radius = float(np.max(min_angles))
    
    return max_radius


def _potential_and_grad(points, power=4.0):
    """Compute the repulsive potential and its gradient.
    
    Potential: sum_{i<j} 1/||p_i - p_j||^power
    This encourages points to spread out.
    """
    n = len(points)
    potential = 0.0
    grad = np.zeros_like(points)
    
    for i in range(n):
        for j in range(i + 1, n):
            d = points[i] - points[j]
            dist_sq = np.dot(d, d)
            if dist_sq < 1e-12:
                # Points are too close, add a large penalty
                potential += 1e10
                continue
            dist = np.sqrt(dist_sq)
            # Potential term: 1/dist^power
            pot = 1.0 / (dist ** power)
            potential += pot
            
            # Gradient with respect to p_i:
            # d/dp_i [1/||p_i - p_j||^power] = -power/||p_i - p_j||^(power+2) * (p_i - p_j)
            # Wait, let's be careful.
            # f = ||p_i - p_j||^(-power)
            # df/dp_i = -power * ||p_i - p_j||^(-power-2) * (p_i - p_j)
            # Actually: d/dp_i ||p_i - p_j||^2 = 2(p_i - p_j)
            # So d/dp_i ||p_i - p_j|| = (p_i - p_j)/||p_i - p_j||
            # d/dp_i ||p_i - p_j||^(-power) = -power * ||p_i - p_j||^(-power-1) * (p_i - p_j)/||p_i - p_j||
            # = -power * ||p_i - p_j||^(-power-2) * (p_i - p_j)
            
            coeff = -power / (dist_sq ** (power / 2.0 + 1.0))
            grad[i] += coeff * d
            grad[j] -= coeff * d  # Gradient with respect to p_j is the negative
    
    return potential, grad


def construct():
    """Construct 50 points on the unit sphere to minimize the covering radius."""
    
    # Start with a good initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Use scipy's minimize with L-BFGS-B for optimization
    # We'll minimize a combination of the repulsive potential and use
    # the covering radius as the objective for final refinement
    
    # First, optimize using the repulsive potential to get a well-spread configuration
    def objective(x):
        pts = x.reshape(N, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        pts = pts / norms
        
        # Compute repulsive potential
        pot, _ = _potential_and_grad(pts, power=4.0)
        return pot
    
    # Use multiple random restarts with different initial conditions
    best_points = points.copy()
    best_pot = objective(points.flatten())
    
    rng = np.random.default_rng(12345)
    
    for restart in range(5):
        if restart == 0:
            x0 = points.copy()
        else:
            # Perturb the current best or use a different spiral
            x0 = best_points.copy()
            noise = rng.normal(0, 0.1, size=(N, 3))
            x0 = x0 + noise
            x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
        
        try:
            result = minimize(
                objective,
                x0.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_pot:
                best_pot = result.fun
                best_points = result.x.reshape(N, 3)
                best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
        except Exception:
            continue
    
    # Final refinement: minimize the covering radius directly
    # Use a local search around the best configuration
    
    def covering_radius_obj(x):
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        pts = pts / norms
        return _covering_radius(pts)
    
    # Try to refine using the covering radius as objective
    try:
        result = minimize(
            covering_radius_obj,
            best_points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        
        if result.fun < covering_radius_obj(best_points.flatten()):
            best_points = result.x.reshape(N, 3)
            best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    except Exception:
        pass
    
    # Ensure the result is valid: nonzero vectors, normalized, not in a closed hemisphere
    norms = np.linalg.norm(best_points, axis=1)
    best_points = best_points / np.maximum(norms, 1e-12)
    
    # Check if all points are in a closed hemisphere
    # This is true if there exists a vector v such that v·p_i >= 0 for all i
    # We can check by seeing if the origin is in the convex hull
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(best_points)
        # If the origin is not in the convex hull, the points span the sphere
        # Check if origin is inside
        # A simpler check: if all points have positive dot product with some direction, they're in a hemisphere
        # Use the fact that points are in a hemisphere iff the convex hull doesn't contain the origin
        # We can check by solving a linear programming problem, but let's use a simpler heuristic
        # If the maximum angular distance from the centroid direction is < 90 degrees, they're in a hemisphere
        centroid = np.sum(best_points, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 1e-10:
            centroid = centroid / centroid_norm
            dots = best_points @ centroid
            if np.all(dots > 0.99):  # All points within ~8 degrees of centroid direction
                # This is a very tight hemisphere, try to fix by flipping some points
                # But this is unlikely with a good optimization
                pass
    except Exception:
        pass
    
    return best_points
# EVOLVE-BLOCK-END
