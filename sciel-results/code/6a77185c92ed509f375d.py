"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

# EVOLVE-BLOCK-START
def _covering_radius(points):
    """Compute the covering radius in radians for a set of points on the unit sphere."""
    try:
        hull = ConvexHull(points)
        max_ang = 0.0
        for simplex in hull.simplices:
            tri = points[simplex]
            # Compute circumcenter of the triangle
            a, b, c = tri[0], tri[1], tri[2]
            # Using the formula for circumcenter of a triangle on the sphere
            # For a spherical triangle, the circumcenter is the normalized sum
            # of the vertices weighted by the opposite edge lengths, but a simpler
            # approach: the circumcenter of the planar triangle projected back.
            # Actually, for the convex hull facet, the circumcenter in 3D is the
            # point equidistant from all three vertices. For a triangle on the unit
            # sphere, the circumcenter (as a direction) can be found by normalizing
            # the vector that is equidistant from all three.
            # The circumcenter of a triangle with vertices a, b, c (unit vectors)
            # in 3D: it's the intersection of the sphere with the plane perpendicular
            # to the triangle's plane passing through the triangle's circumcenter.
            # Simpler: compute the circumcenter of the planar triangle, then normalize.
            # But that's not quite right for spherical geometry.
            # The correct approach: the circumcenter of the spherical triangle is the
            # point on the sphere equidistant (angularly) from a, b, c.
            # This is the normalized sum of a, b, c weighted appropriately, or we can
            # solve the system.
            # For a triangle on the unit sphere, the circumcenter direction is:
            # normalize(a + b + c) is NOT generally correct.
            # The correct formula: the circumcenter is proportional to
            # a*(|b-c|^2) + b*(|c-a|^2) + c*(|a-b|^2) ... no, that's for planar.
            # Let me use the formula: circumcenter = (a + b + c) / |a + b + c| is wrong.
            # The actual circumcenter of a spherical triangle: it's the point where
            # the perpendicular bisectors meet. In 3D, for a triangle on the unit sphere,
            # the circumcenter (as a unit vector) is:
            # n = (a × b) × (b × c) normalized? No.
            # Let me think again. The circumcenter of a triangle with vertices on the unit
            # sphere is the unit vector u such that u·a = u·b = u·c.
            # This means u is in the null space of [a-b; b-c] (as a linear system).
            # u · (a - b) = 0 and u · (b - c) = 0, and |u| = 1.
            # So u is parallel to (a-b) × (b-c).
            v1 = a - b
            v2 = b - c
            u = np.cross(v1, v2)
            if np.linalg.norm(u) < 1e-15:
                # Degenerate triangle
                u = a + b + c
            u = u / np.linalg.norm(u)
            # The angular distance from u to any vertex
            ang = np.arccos(np.clip(np.dot(u, a), -1.0, 1.0))
            if ang > max_ang:
                max_ang = ang
        return max_ang
    except Exception:
        return np.pi


def _refine(points, max_time=140.0):
    """Refine points using gradient-based optimization to minimize covering radius."""
    start_time = time.time()
    points = points.copy()
    
    # Use a repulsive force / optimization approach
    # We'll use scipy's minimize with a smooth objective
    
    def objective(x):
        """Objective: covering radius computed via convex hull."""
        pts = x.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        return _covering_radius(pts)
    
    # Gradient-free optimization using Nelder-Mead or Powell
    # Since the objective is non-smooth, use a derivative-free method
    best_x = points.flatten()
    best_f = objective(best_x)
    
    # Use Powell's method which handles non-smooth objectives reasonably
    options = {'maxiter': 200, 'ftol': 1e-10, 'xtol': 1e-10}
    
    while time.time() - start_time < max_time:
        res = minimize(objective, best_x, method='Powell', options=options)
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
        else:
            # Try a different starting point with small perturbation
            break
    
    # Normalize final result
    pts = best_x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    pts = pts / norms
    return pts


def construct():
    """Construct 130 points on the unit sphere minimizing covering radius."""
    start_time = time.time()
    
    # Golden-angle (Fibonacci) spiral points on the unit sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Refine using optimization
    if time.time() - start_time < 130.0:
        points = _refine(points, max_time=130.0)
    
    # Final normalization
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    points = points / norms
    
    return points
# EVOLVE-BLOCK-END
