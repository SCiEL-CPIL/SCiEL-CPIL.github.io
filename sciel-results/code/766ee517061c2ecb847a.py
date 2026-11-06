"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import math

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _normalize(points):
    """Normalize points to unit length."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return points / norms

def _covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    The covering radius is the maximum over all facets of the convex hull
    of the circumradius of that facet (in angular distance).
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi / 2  # fallback: hemisphere

    max_radius = 0.0
    n = len(points)
    
    for simplex in hull.simplices:
        # Get the points defining this facet
        facet_points = points[simplex]
        d = len(simplex)  # dimension of the simplex (facet is (d-1)-dim)
        
        if d == 1:
            # Edge: the farthest point is the antipode of the midpoint
            mid = (facet_points[0] + facet_points[1]) / 2.0
            mid_norm = np.linalg.norm(mid)
            if mid_norm < 1e-10:
                continue
            # The covering radius for an edge is pi - angle_between(points) / 2
            # Actually, for a single edge, the farthest point on the sphere
            # from both endpoints is at the antipode of the normalized midpoint.
            # The angular distance from the midpoint (normalized) to either endpoint
            # is the half-angle. The covering radius is pi - half_angle? No.
            # Let's think: the cap is defined by the plane through the facet.
            # For an edge, the "cap" is a lune. The farthest point is the antipode
            # of the normalized midpoint. The distance from that point to an endpoint
            # is pi - angle(endpoint, midpoint_normalized).
            # But actually, the covering radius is the maximum distance to the nearest point.
            # For a facet, the farthest point in the cap is the circumcenter of the facet
            # (extended to the sphere). If the circumcenter is outside the facet,
            # the maximum is at a vertex or edge.
            
            # A simpler approach for the covering radius:
            # For each facet, compute the circumcenter of the facet points on the sphere.
            # The circumcenter is the point on the sphere that is equidistant from all facet points.
            # If the circumcenter lies in the "interior" of the facet (i.e., the facet is acute),
            # then the covering radius for that facet is the distance from the circumcenter to any facet point.
            # Otherwise, the maximum is achieved at a lower-dimensional face.
            
            # For robustness, let's compute the circumcenter of the facet.
            pass
        
        # Compute circumcenter of the facet
        # For a set of points on the unit sphere, the circumcenter is the normalized solution
        # to the linear system: (p_i - p_0) . x = 0 for all i, and x is on the unit sphere.
        # This is equivalent to finding the null space of the matrix with rows (p_i - p_0).
        
        if d < 2:
            continue
            
        # Build the linear system
        p0 = facet_points[0]
        A = facet_points[1:] - p0  # shape (d-1, 3)
        
        # Find the null space of A
        try:
            _, _, vh = np.linalg.svd(A)
            circumcenter = vh[-1]  # last row of vh
        except np.linalg.LinAlgError:
            continue
        
        circumcenter = circumcenter / np.linalg.norm(circumcenter)
        
        # Check if the circumcenter is "inside" the facet
        # For a triangle on the sphere, this means the circumcenter is in the same hemisphere
        # as the facet normal.
        # Compute the facet normal
        if d == 2:
            normal = np.cross(facet_points[1] - facet_points[0], facet_points[2] - facet_points[0])
        elif d == 3:
            normal = np.cross(facet_points[1] - facet_points[0], facet_points[2] - facet_points[0])
            # For a tetrahedron, we need to check all four faces
            pass
        else:
            normal = np.cross(facet_points[1] - facet_points[0], facet_points[2] - facet_points[0])
        
        normal = normal / np.linalg.norm(normal)
        
        # The circumcenter should be on the same side as the normal
        # Actually, for the covering radius, we want the point on the sphere that is
        # farthest from all facet points but still in the cap.
        # The circumcenter is the unique point equidistant from all facet points.
        # If it's in the cap, it's the covering point.
        
        # The angular distance from the circumcenter to any facet point
        dot = np.clip(np.dot(circumcenter, p0), -1.0, 1.0)
        angle = np.arccos(dot)
        
        # But we need to check if the circumcenter is actually in the facet's cap.
        # A simpler check: the covering radius for the facet is the maximum over
        # all points in the facet of the distance to the nearest point.
        # This is achieved at the circumcenter if it's inside, otherwise at a vertex/edge.
        
        # For now, let's just use the circumcenter distance as an upper bound.
        # This is not always correct, but it's a reasonable approximation.
        
        max_radius = max(max_radius, angle)
    
    return max_radius

def _soft_covering_radius(points, beta=50.0):
    """
    A smooth surrogate for the covering radius using log-sum-exp.
    This is differentiable and can be used with gradient-based optimization.
    """
    # For each pair of points, compute the angular distance
    # The covering radius is the maximum over all facets, but as a surrogate,
    # we can use the maximum over all pairs of points.
    # This is not exactly the covering radius, but it's a smooth approximation.
    
    n = len(points)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dot = np.clip(np.dot(points[i], points[j]), -1.0, 1.0)
            distances[i, j] = np.arccos(dot)
    
    # The covering radius is approximately the maximum over all pairs
    # But we want a smooth version. Use log-sum-exp.
    max_dist = np.max(distances)
    if max_dist < 1e-10:
        return 0.0
    
    # Soft max
    scaled = distances / beta
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return log_sum_exp * beta

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Start with Fibonacci spiral
    points = _fibonacci_sphere(N)
    
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    # Use L-BFGS-B to minimize the soft covering radius
    # We need to parameterize the sphere. Use spherical coordinates.
    # Or use the fact that we can project back to the sphere.
    
    def objective(x):
        """Objective function: soft covering radius."""
        pts = _normalize(x)
        return _soft_covering_radius(pts, beta=30.0)
    
    def gradient(x):
        """Gradient of the objective function."""
        pts = _normalize(x)
        n = len(pts)
        
        # Compute the gradient of the soft covering radius
        # This is complex, so let's use finite differences as a fallback
        # or use a simpler gradient.
        
        # For now, let's use a simple gradient approximation
        eps = 1e-6
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (objective(x_plus) - objective(x)) / eps
        
        return grad
    
    # Try optimization with multiple restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x0 = points.flatten()
        else:
            # Add some random perturbation
            x0 = points.flatten() + 0.1 * np.random.randn(len(points.flatten()))
        
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            
            if result.success or result.fun < objective(x0):
                new_points = _normalize(result.x.reshape(N, 3))
                new_radius = _covering_radius(new_points)
                
                if new_radius < best_radius:
                    best_points = new_points
                    best_radius = new_radius
                    
        except Exception as e:
            continue
    
    # Final polish: try to improve with a local search
    if time.time() < deadline - 5.0:
        # Simple local search: perturb each point slightly and keep if it improves
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(N):
                for _ in range(10):
                    # Random small perturbation
                    delta = 0.01 * np.random.randn(3)
                    new_point = best_points[i] + delta
                    new_point = new_point / np.linalg.norm(new_point)
                    
                    old_points = best_points.copy()
                    best_points[i] = new_point
                    new_radius = _covering_radius(best_points)
                    
                    if new_radius < best_radius:
                        best_radius = new_radius
                        improved = True
                    else:
                        best_points[i] = old_points[i]
    
    return _normalize(best_points)

# EVOLVE-BLOCK-END
