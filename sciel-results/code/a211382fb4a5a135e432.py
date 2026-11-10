"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _covering_radius(points):
    """Compute the covering radius in degrees for points on the unit sphere."""
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        # For each facet, compute the circumradius
        # A facet is a triangle (3 points) or higher-dimensional simplex
        # For a triangle with vertices a, b, c on unit sphere:
        # circumradius R satisfies: R = |a||b||c| / (2 * area)
        # But more robustly: find the center of the circumscribed sphere of the facet
        # and compute the angular distance from that center to any vertex
        
        max_circumradius = 0.0
        for simplex in facets:
            pts = points[simplex]
            n_pts = len(pts)
            if n_pts < 3:
                continue
            # Center of circumscribed sphere for the simplex
            # For a triangle: solve for point equidistant from all 3 vertices
            # General approach: use the fact that the circumcenter of a facet
            # on the unit sphere lies along the normal of the facet
            # 
            # More direct: for a triangle with vertices a, b, c on unit sphere,
            # the circumradius (angular) is the maximum angle from the facet center
            # to any vertex. The facet center is the point on the unit sphere
            # that is equidistant from all facet vertices.
            
            # For a triangle, the circumcenter can be found as follows:
            # The plane through a, b, c has normal n = (b-a) x (c-a)
            # The circumcenter of the triangle (in 3D, on the unit sphere) is
            # the intersection of the unit sphere with the line through the
            # triangle's centroid in the direction of the normal... 
            # Actually, let's use a more robust method.
            
            # Compute the circumradius directly:
            # For a spherical triangle, the circumradius R satisfies:
            # sin(R) = |a x b| * |b x c| * |c x a| / (4 * |a x b x c|) for 3 points
            # This is the formula for the circumradius of a spherical triangle.
            
            if n_pts == 3:
                a, b, c = pts[0], pts[1], pts[2]
                ab = np.cross(a, b)
                bc = np.cross(b, c)
                ca = np.cross(c, a)
                triple = np.dot(a, np.cross(b, c))
                denom = 4.0 * abs(triple)
                if denom < 1e-15:
                    continue
                sin_R = (np.linalg.norm(ab) * np.linalg.norm(bc) * np.linalg.norm(ca)) / denom
                sin_R = min(sin_R, 1.0)
                R = np.arcsin(sin_R)
                max_circumradius = max(max_circumradius, R)
            elif n_pts == 4:
                # Tetrahedron facet: compute circumradius
                a, b, c, d = pts[0], pts[1], pts[2], pts[3]
                # For a tetrahedron on the unit sphere, the circumradius is the
                # angular distance from the circumcenter to any vertex.
                # The circumcenter is the point equidistant from all 4 vertices.
                # Solve: |x - a|^2 = |x - b|^2 = |x - c|^2 = |x - d|^2
                # This gives 3 linear equations.
                # 2*x·(a-b) = |a|^2 - |b|^2 = 0 (since all on unit sphere)
                # So x·(a-b) = 0, x·(a-c) = 0, x·(a-d) = 0
                # x is orthogonal to (a-b), (a-c), (a-d)
                # x = (a-b) x ((a-c) x (a-d))
                v1 = a - b
                v2 = a - c
                v3 = a - d
                x = np.cross(v1, np.cross(v2, v3))
                nx = np.linalg.norm(x)
                if nx < 1e-15:
                    continue
                x = x / nx
                # Angular distance from x to a
                cos_R = np.clip(np.dot(x, a), -1.0, 1.0)
                R = np.arccos(cos_R)
                max_circumradius = max(max_circumradius, R)
            else:
                # Higher-dimensional facet: use centroid as approximation
                centroid = np.mean(pts, axis=0)
                nc = np.linalg.norm(centroid)
                if nc < 1e-15:
                    continue
                centroid = centroid / nc
                # Max angular distance to any vertex
                for p in pts:
                    cos_R = np.clip(np.dot(centroid, p), -1.0, 1.0)
                    R = np.arccos(cos_R)
                    max_circumradius = max(max_circumradius, R)
        
        return np.degrees(max_circumradius)
    except Exception:
        return np.inf


def _optimize_from_start(points, max_time=100.0):
    """Local optimization using gradient-free methods."""
    start_time = time.time()
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    # Use a simple hill-climbing / Nelder-Mead-like approach
    # We'll use coordinate-wise perturbations
    
    step_size = 0.01
    n_iter = 0
    while time.time() - start_time < max_time:
        n_iter += 1
        improved = False
        for i in range(N):
            if time.time() - start_time > max_time:
                break
            for dim in range(3):
                # Try small perturbations in each direction
                for delta in [step_size, -step_size]:
                    candidate = best_points.copy()
                    candidate[i, dim] += delta
                    # Renormalize
                    candidate[i] = candidate[i] / np.linalg.norm(candidate[i])
                    r = _covering_radius(candidate)
                    if r < best_radius:
                        best_points = candidate
                        best_radius = r
                        improved = True
                        break
        if not improved:
            step_size *= 0.8
            if step_size < 1e-6:
                break
    
    return best_points, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points
    best_radius = _covering_radius(best_points)
    
    # Try multiple random restarts with local optimization
    np.random.seed(42)
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random start
        rand_points = np.random.randn(N, 3)
        rand_points /= np.linalg.norm(rand_points, axis=1, keepdims=True)
        
        # Optimize from this start
        opt_points, opt_radius = _optimize_from_start(rand_points, max_time=20.0)
        
        if opt_radius < best_radius:
            best_points = opt_points
            best_radius = opt_radius
    
    # Final optimization from the best point found
    if time.time() < deadline:
        final_points, final_radius = _optimize_from_start(best_points, max_time=30.0)
        if final_radius < best_radius:
            best_points = final_points
            best_radius = final_radius
    
    return best_points
# EVOLVE-BLOCK-END
