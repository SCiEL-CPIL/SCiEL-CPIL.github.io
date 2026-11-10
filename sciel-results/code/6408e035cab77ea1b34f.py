"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _compute_facet_data(points):
    """
    Compute the circumradius and circumcenter for each facet of the convex hull.
    Returns:
        hull: The ConvexHull object
        circumradii: Array of circumradii (in radians) for each facet
        circumcenters: Array of circumcenters (unit vectors) for each facet
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, np.array([]), np.array([])
    
    facets = hull.simplices
    n_facets = len(facets)
    circumradii = np.zeros(n_facets)
    circumcenters = np.zeros((n_facets, 3))
    
    # We process facets in batches or one by one, but vectorizing the math inside is key.
    # For a triangle (a,b,c), the circumcenter on the sphere is along the normal.
    # Normal n = (b-a) x (c-a)
    # The circumcenter C is the intersection of the sphere and the line through the centroid? No.
    # For a spherical triangle, the circumcenter is the pole of the great circle? No.
    # It is the point equidistant from a, b, c.
    # Let C = alpha * n. Since C is on the unit sphere, |C|=1.
    # C dot a = C dot b = C dot c = cos(R)
    # (alpha * n) dot a = cos(R)
    # alpha * (n dot a) = cos(R)
    # So alpha = cos(R) / (n dot a)
    # We need to find R.
    # sin(R) = |a x b| |b x c| |c x a| / (4 * |a x (b x c)|)
    # This is computationally heavy per facet.
    
    # Alternative: For each facet, solve for the circumcenter directly.
    # For a triangle a, b, c:
    # C is perpendicular to (a-b), (b-c), (c-a)? No, C is in the plane of the triangle? No, C is on the sphere.
    # The vector C is in the span of a, b, c.
    # Actually, C is parallel to the normal of the plane containing a, b, c?
    # Yes, for a triangle on the unit sphere, the circumcenter is the normalized normal vector to the plane passing through a, b, c.
    # Let's verify. If C = n / |n|, then C dot a = C dot b = C dot c?
    # n = (b-a) x (c-a)
    # n dot a = ((b-a) x (c-a)) dot a = (b-a) x (c-a) dot a
    # This is the triple product a . (b-a) x (c-a) = a . (b x c - b x a - a x c + a x a)
    # = a . (b x c) - 0 - 0 + 0 = a . (b x c)
    # Similarly n dot b = b . (c x a) = b . (c x a) = (b, c, a) triple product = (a, b, c) triple product.
    # n dot c = c . (a x b) = (c, a, b) triple product = (a, b, c) triple product.
    # So yes, n dot a = n dot b = n dot c = T (the triple product scalar).
    # Therefore, if we set C = n / |n|, then C dot a = T / |n|.
    # This is constant for all three vertices. So C is equidistant from a, b, c.
    # And C is on the unit sphere.
    # So the circumcenter of a triangular facet is simply the normalized normal vector.
    
    # For a tetrahedron (a,b,c,d):
    # The circumcenter is the point equidistant from a,b,c,d.
    # x . (a-b) = 0, x . (a-c) = 0, x . (a-d) = 0.
    # x is orthogonal to the differences.
    # x = (a-b) x ((a-c) x (a-d))
    
    for i in range(n_facets):
        idx = facets[i]
        pts = points[idx]
        n_pts = len(idx)
        
        if n_pts == 3:
            a, b, c = pts
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-15:
                circumradii[i] = np.inf
                continue
            C = n / nn
            # Ensure C is on the same side as the vertices (dot product positive)
            # If C dot a < 0, the "cap" is on the other side, but the circumradius is the angle.
            # The angle is arccos(|C dot a|)? No, the covering radius is the distance to the nearest point.
            # The circumcenter of the facet *on the sphere* is the point that maximizes the distance to the vertices?
            # No, the circumcenter is equidistant. The distance is R.
            # The "hole" is centered at C. The radius of the hole is R.
            # We want to minimize R.
            cos_R = np.dot(C, a)
            # If cos_R is negative, the center is on the opposite side.
            # For a convex hull, the normal points outward. The vertices are on the sphere.
            # The angle between C and a is the circumradius.
            # If the facet is "flat", C is close to the centroid.
            circumradii[i] = np.arccos(np.clip(cos_R, -1.0, 1.0))
            circumcenters[i] = C
            
        elif n_pts == 4:
            a, b, c, d = pts
            v1 = a - b
            v2 = a - c
            v3 = a - d
            x = np.cross(v1, np.cross(v2, v3))
            nx = np.linalg.norm(x)
            if nx < 1e-15:
                circumradii[i] = np.inf
                continue
            C = x / nx
            cos_R = np.dot(C, a)
            circumradii[i] = np.arccos(np.clip(cos_R, -1.0, 1.0))
            circumcenters[i] = C
        else:
            # Higher dimensional: use centroid approximation
            centroid = np.mean(pts, axis=0)
            nc = np.linalg.norm(centroid)
            if nc < 1e-15:
                circumradii[i] = np.inf
                continue
            C = centroid / nc
            cos_R = np.dot(C, a)
            circumradii[i] = np.arccos(np.clip(cos_R, -1.0, 1.0))
            circumcenters[i] = C
            
    return hull, circumradii, circumcenters


def _covering_radius(points):
    """Compute the covering radius in degrees."""
    _, circumradii, _ = _compute_facet_data(points)
    if len(circumradii) == 0:
        return np.inf
    return np.degrees(np.max(circumradii))


def _optimize_worst_facet(points, max_time=60.0):
    """
    Optimize by repeatedly finding the worst facet and pushing its vertices 
    towards its circumcenter to reduce the circumradius.
    """
    start_time = time.time()
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    step_size = 0.1
    no_improve_count = 0
    
    while time.time() - start_time < max_time:
        hull, circumradii, circumcenters = _compute_facet_data(best_points)
        if len(circumradii) == 0:
            break
            
        worst_idx = np.argmax(circumradii)
        worst_radius = circumradii[worst_idx]
        
        if worst_radius <= best_radius + 1e-10:
            # No improvement possible? Or we are stuck?
            # Actually, best_radius is the max. So worst_radius should be >= best_radius.
            # If we improved, best_radius would have updated.
            pass
            
        # Get the worst facet's vertices
        worst_facet = hull.simplices[worst_idx]
        C = circumcenters[worst_idx]
        
        # Push vertices towards C
        # Spherical linear interpolation (slerp) or simple vector addition and renormalization
        # Move v towards C by step_size
        # v_new = (v + step_size * C) / |v + step_size * C|
        # This moves v in the direction of C on the sphere.
        
        improved = False
        for i in worst_facet:
            v = best_points[i]
            # Check if v is already at C
            if np.linalg.norm(v - C) < 1e-8:
                continue
            
            v_new = v + step_size * C
            norm_v_new = np.linalg.norm(v_new)
            if norm_v_new < 1e-8:
                continue
            v_new = v_new / norm_v_new
            
            candidate = best_points.copy()
            candidate[i] = v_new
            
            # Evaluate only the change? No, just recompute.
            # To speed up, we could check if this move reduces the local circumradius of the worst facet,
            # but global changes might break other facets.
            # Given N=130, recompute is ~1-5ms. 1000 iterations in 5s is fine.
            
            new_radius = _covering_radius(candidate)
            
            if new_radius < best_radius:
                best_points = candidate
                best_radius = new_radius
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        if not improved:
            step_size *= 0.9
            if step_size < 1e-6:
                break
            # If stuck, try a smaller step or different facet?
            # For now, just reduce step.
            
    return best_points, best_radius


def _polish(points, max_time=20.0):
    """
    High-precision polish phase with adaptive step-size reduction.
    """
    start_time = time.time()
    best_points = points.copy()
    best_radius = _covering_radius(best_points)
    
    base_alpha = 0.01
    
    while time.time() - start_time < max_time:
        hull, circumradii, circumcenters = _compute_facet_data(best_points)
        if len(circumradii) == 0:
            break
            
        worst_idx = np.argmax(circumradii)
        worst_radius = circumradii[worst_idx]
        
        if worst_radius < 1e-10:
            break
            
        worst_facet = hull.simplices[worst_idx]
        C = circumcenters[worst_idx]
        
        improved = False
        # Try multiple step sizes
        for alpha_mult in [1.0, 0.5, 0.25, 0.1]:
            alpha = base_alpha * alpha_mult
            for i in worst_facet:
                v = best_points[i]
                v_new = v + alpha * C
                norm_v_new = np.linalg.norm(v_new)
                if norm_v_new < 1e-8:
                    continue
                v_new = v_new / norm_v_new
                
                candidate = best_points.copy()
                candidate[i] = v_new
                
                new_radius = _covering_radius(candidate)
                
                if new_radius < best_radius:
                    best_points = candidate
                    best_radius = new_radius
                    improved = True
                    break # Try next facet or restart alpha loop?
            if improved:
                break
                
        if not improved:
            base_alpha *= 0.5
            if base_alpha < 1e-8:
                break
                
    return best_points, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points
    best_radius = _covering_radius(best_points)
    
    # Try a few random restarts to escape local minima
    np.random.seed(42)
    n_restarts = 3
    time_per_restart = (deadline - time.time()) / (n_restarts + 1)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0: # Leave 30s for final polish
            break
            
        rand_points = np.random.randn(N, 3)
        rand_points /= np.linalg.norm(rand_points, axis=1, keepdims=True)
        
        # Optimize from this start
        opt_points, opt_radius = _optimize_worst_facet(rand_points, max_time=time_per_restart)
        
        if opt_radius < best_radius:
            best_points = opt_points
            best_radius = opt_radius
            
    # Final optimization from the best point found
    if time.time() < deadline - 20.0:
        final_points, final_radius = _optimize_worst_facet(best_points, max_time=20.0)
        if final_radius < best_radius:
            best_points = final_points
            best_radius = final_radius
            
    # Polish phase
    if time.time() < deadline - 5.0:
        polished_points, polished_radius = _polish(best_points, max_time=15.0)
        if polished_radius < best_radius:
            best_points = polished_points
            best_radius = polished_radius
            
    return best_points
# EVOLVE-BLOCK-END
