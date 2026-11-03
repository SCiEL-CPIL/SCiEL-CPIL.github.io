"""Optimized sphere covering for 130 points."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _covering_radius_deg(pts):
    """Compute covering radius in degrees for a set of unit vectors."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 90.0
    max_r = 0.0
    for simplex in hull.simplices:
        v = pts[simplex]  # shape (k, 3)
        # Circumcenter of the simplex
        # Solve for c such that ||c - v_i||^2 = R^2 for all i
        # This gives linear equations: 2*c dot (v_i - v_0) = ||v_i||^2 - ||v_0||^2
        # Since all on unit sphere, ||v_i||^2 = 1, so RHS = 0
        # So 2*c dot (v_i - v_0) = 0 for i=1..k-1
        # We need c to be equidistant from all vertices and lie on the sphere outward from centroid
        centroid = v.mean(axis=0)
        # The circumcenter of a spherical cap facet lies along the direction of the centroid
        # normalized, at some distance. Actually for points on unit sphere, the circumcenter
        # of the facet (as a planar triangle) projected... 
        # Better: the covering radius for a facet is the angular distance from the facet's
        # "center" (the point on the sphere farthest from all vertices in that cap) to the vertices.
        # For a spherical triangle with vertices v1,v2,v3, the point farthest from all three
        # within the triangle is the circumcenter of the spherical triangle.
        # The circumcenter c (unit vector) satisfies c dot v_i = cos(r) for all i.
        # So c is equidistant (in dot product) from all vertices.
        # Solve: find c on unit sphere minimizing max_i angle(c, v_i)
        # Equivalently, c is the point on the sphere that is equidistant from all v_i
        # (if the circumcenter exists inside the triangle).
        # For a triangle, the circumcenter satisfies c dot (v_i - v_j) = 0 for all pairs.
        # So c is orthogonal to all edge vectors, i.e., c is parallel to the cross product
        # of two edges, which is the normal to the plane through the three points.
        # But that's the planar circumcenter direction. For spherical geometry, the circumcenter
        # of the spherical triangle is indeed along the direction of the normal to the plane
        # containing the three points, IF the triangle is acute. Let's just compute it numerically.
        
        # Simple approach: the circumcenter of the spherical triangle is the point c on the unit
        # sphere such that c·v_i is constant for all i. This means c is in the null space of
        # the differences. For 3 points, c is proportional to the cross product of (v1-v2) and (v1-v3),
        # i.e., the normal to the plane. But we need to pick the correct sign and normalize.
        # Actually, the point on the unit sphere equidistant from v1,v2,v3 is along the direction
        # of the vector that is the "spherical circumcenter". Let's just use the fact that
        # for three unit vectors, the circumcenter direction is:
        # c = normalize( v1 × v2 × v3 )? No.
        # Let's solve the linear system. We want c·v_i = d for all i.
        # c·(v_i - v_0) = 0 for i=1,2
        # So c is orthogonal to (v1-v0) and (v2-v0), hence c is parallel to (v1-v0)×(v2-v0).
        n = np.cross(v[1] - v[0], v[2] - v[0])
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            # Degenerate, use centroid direction
            c = centroid / np.linalg.norm(centroid)
        else:
            c = n / n_norm
            # Choose sign so that c is on the same side as the centroid (outward)
            if np.dot(c, centroid) < 0:
                c = -c
        # The angular distance from c to any vertex is the covering radius for this facet
        r = np.arccos(np.clip(np.dot(c, v[0]), -1.0, 1.0))
        max_r = max(max_r, r)
    return np.degrees(max_r)


def _gradient(pts):
    """Compute gradient of covering radius with respect to point positions."""
    # Use finite differences for simplicity, but this is expensive.
    # Alternative: use the fact that only the facets with max radius contribute.
    # For now, use a simple subgradient approach.
    pass


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_r = _covering_radius_deg(pts)
    
    # Local refinement using L-BFGS-B with finite differences
    # We optimize the positions on the sphere by parameterizing with tangent vectors
    # Or simply optimize in R^3 and project to sphere
    
    def objective(x):
        p = x.reshape(N, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        p = p / norms
        return _covering_radius_deg(p)
    
    # Use multiple random restarts and local optimization
    rng = np.random.RandomState(42)
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        # Perturb the initial points
        if restart == 0:
            x0 = best_pts.flatten()
        else:
            x0 = best_pts + rng.randn(N, 3) * 0.1
            # Normalize
            norms = np.linalg.norm(x0.reshape(N, 3), axis=1, keepdims=True)
            x0 = (x0.reshape(N, 3) / norms).flatten()
        
        # Optimize
        try:
            res = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 5000}
            )
            if res.success or res.fun < best_r:
                p_new = res.x.reshape(N, 3)
                norms = np.linalg.norm(p_new, axis=1, keepdims=True)
                p_new = p_new / np.maximum(norms, 1e-10)
                r_new = _covering_radius_deg(p_new)
                if r_new < best_r:
                    best_r = r_new
                    best_pts = p_new
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5:
        try:
            res = minimize(
                objective,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 10000, 'gtol': 1e-8}
            )
            if res.fun < best_r:
                p_new = res.x.reshape(N, 3)
                norms = np.linalg.norm(p_new, axis=1, keepdims=True)
                p_new = p_new / np.maximum(norms, 1e-10)
                best_r = _covering_radius_deg(p_new)
                best_pts = p_new
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
