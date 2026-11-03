"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # safety margin before 150s kill

def _to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius_and_facets(pts):
    """
    Compute the covering radius (in radians) and identify the critical facet.
    Returns (radius, facet_indices, circumcenter_direction).
    """
    pts = _to_sphere(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback if hull fails (should not happen for generic position)
        return _covering_radius_fallback(pts), None, None
    
    max_radius = 0.0
    best_facet = None
    best_center = None
    
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Circumcenter direction
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            continue
        
        x = cross / norm_cross
        
        # Angular distance from x to vertex a
        dot_a = np.dot(x, a)
        dot_a = np.clip(dot_a, -1.0, 1.0)
        r1 = np.arccos(dot_a)
        
        # The other direction
        dot_a_neg = -dot_a
        dot_a_neg = np.clip(dot_a_neg, -1.0, 1.0)
        r2 = np.arccos(dot_a_neg)
        
        # Take the larger circumradius (the one in the "empty" cap direction)
        candidate = max(r1, r2)
        if candidate > max_radius:
            max_radius = candidate
            best_facet = simplex
            # Determine which direction is the "empty" one
            # The center of the empty cap is the one furthest from all points
            # Actually, the circumcenter of the triangle on the sphere is unique up to sign.
            # We need the one that is NOT "inside" the polytope.
            # The normal x points away from the origin if the triangle is on the "front".
            # Let's check distance to all points. The covering center is the point on the sphere
            # equidistant to a,b,c and maximally far from others.
            # For a facet of the convex hull, the outward normal is the direction of the empty cap.
            # The outward normal of the facet defined by a,b,c in CCW order (from outside) 
            # is cross(b-a, c-a). 
            # ConvexHull simplices are ordered such that the normal points outward.
            # So x is the direction of the empty cap center.
            best_center = x
            
    # Check antipodes of each point
    # The covering radius might also be determined by the antipode of a point
    # if the point is isolated enough.
    # Specifically, for each point p, the farthest point in its Voronoi cell might be -p?
    # No, -p is the farthest point on the sphere from p.
    # The distance from -p to the nearest configuration point is the distance from -p to the set P.
    # If -p is in the Voronoi cell of some point q, then the distance is d(-p, q).
    # But if -p is not in any cell? It must be in some cell.
    # The covering radius is max over y of min_i d(y, p_i).
    # The candidates for y are circumcenters of facets and antipodes of points.
    
    min_dist_to_anti = 0.0
    for i in range(len(pts)):
        neg_pt = -pts[i]
        dots = pts @ neg_pt
        dots = np.clip(dots, -1.0, 1.0)
        min_d = np.arccos(np.min(dots))
        if min_d > max_radius:
            max_radius = min_d
            # In this case, the "facet" is effectively the antipode
            best_facet = None 
            best_center = neg_pt

    return max_radius, best_facet, best_center

def _covering_radius_fallback(pts):
    """Fallback covering radius computation via moderate sampling."""
    n_theta = 100
    n_phi = 200
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    max_radius = 0.0
    for theta in thetas:
        st = np.sin(theta)
        ct = np.cos(theta)
        y_batch = np.column_stack([
            st * np.cos(phis),
            st * np.sin(phis),
            np.full_like(phis, ct)
        ])
        dots = pts @ y_batch.T
        dots = np.clip(dots, -1.0, 1.0)
        min_dists = np.arccos(np.min(dots, axis=0))
        max_radius = max(max_radius, np.max(min_dists))
    return max_radius

def _objective(pts):
    """Compute the objective (covering radius in radians)."""
    r, _, _ = _covering_radius_and_facets(pts)
    return r

def _spherical_lloyd_iteration(pts):
    """
    Perform one iteration of Spherical Lloyd's algorithm.
    Moves each point to the spherical centroid of its Voronoi cell.
    """
    pts = _to_sphere(pts.copy())
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return pts

    # Map each point to the facets that contain it
    point_to_facets = {i: [] for i in range(N)}
    
    for idx, simplex in enumerate(hull.simplices):
        for i in simplex:
            point_to_facets[i].append(idx)

    new_pts = np.zeros_like(pts)
    
    for i in range(N):
        facets = point_to_facets[i]
        if not facets:
            new_pts[i] = pts[i]
            continue
            
        # The Voronoi cell of point i on the sphere is the intersection of the hemispheres
        # centered at i containing the adjacent facets.
        # A simpler approximation for the centroid:
        # The spherical centroid of the Voronoi cell is the normalized sum of the normals of the facets
        # adjacent to i? No, that's the gradient of the area.
        
        # A robust method for spherical Voronoi centroid:
        # The cell is a spherical polygon. Its vertices are the circumcenters of the facets
        # adjacent to i.
        
        # Let's collect the circumcenters of the adjacent facets.
        centers = []
        for f_idx in facets:
            simplex = hull.simplices[f_idx]
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            v1 = a - b
            v2 = a - c
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            if norm_cross > 1e-12:
                x = cross / norm_cross
                # Ensure x is on the "outside" (away from origin relative to the triangle)
                # The normal from ConvexHull points outward.
                # We want the center of the cell, which is in the direction of the outward normal.
                # However, the circumcenter on the sphere is x.
                # Is x the correct vertex of the Voronoi cell?
                # Yes, the vertex of the Voronoi cell at the intersection of cells a,b,c is the circumcenter.
                centers.append(x)
        
        if not centers:
            new_pts[i] = pts[i]
            continue
        
        centers = np.array(centers)
        
        # The spherical centroid of a set of points on the sphere is the normalized sum
        # if the points are close enough (within a hemisphere).
        # Since these are vertices of the Voronoi cell surrounding i, they should be close to i.
        # However, if the cell is large, they might wrap around.
        # A better centroid: minimize sum of squared distances.
        # For small cells, normalized sum is a good approximation.
        # Let's use the normalized sum of the centers.
        centroid = np.sum(centers, axis=0)
        norm_centroid = np.linalg.norm(centroid)
        if norm_centroid > 1e-12:
            new_pts[i] = centroid / norm_centroid
        else:
            new_pts[i] = pts[i]
            
    return _to_sphere(new_pts)

def _local_optimize_lloyd(pts, max_time):
    """
    Local optimization using Spherical Lloyd's algorithm.
    """
    pts = _to_sphere(pts.copy())
    current_obj = _objective(pts)
    best_pts = pts.copy()
    best_obj = current_obj
    
    start = time.time()
    no_improve = 0
    dampening = 1.0
    
    while time.time() - start < max_time:
        new_pts = _spherical_lloyd_iteration(pts)
        
        # Dampening to prevent oscillation
        pts_damped = (1 - dampening) * pts + dampening * new_pts
        pts_damped = _to_sphere(pts_damped)
        
        new_obj = _objective(pts_damped)
        
        if new_obj < current_obj:
            pts = pts_damped
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_pts = pts.copy()
            no_improve = 0
            dampening = min(dampening * 1.05, 1.0)
        else:
            no_improve += 1
            dampening *= 0.9
            if no_improve > 20:
                break
    
    return best_pts, best_obj

def _gradient_step(pts, step_size):
    """
    Perform a gradient descent step on the covering radius.
    Gradient of the covering radius w.r.t. point i:
    If the covering center is c (circumcenter of facet or antipode),
    and i is one of the points defining c (or the point whose antipode is c),
    the gradient is related to the direction from i to c.
    """
    r, facet, center = _covering_radius_and_facets(pts)
    if center is None:
        return pts
        
    grad = np.zeros_like(pts)
    
    if facet is not None:
        # Center is circumcenter of facet
        for i in facet:
            # Move i towards center to reduce distance?
            # The distance is d(center, p_i).
            # We want to minimize d(center, p_i).
            # The gradient of d(center, p_i) w.r.t p_i is - (center - p_i) / sin(d) ?
            # Actually, moving p_i towards center reduces the distance.
            # So we move p_i in the direction of center.
            # But we must stay on the sphere.
            # Tangential component of center relative to p_i.
            # v = center - (center . p_i) p_i
            proj = np.dot(center, pts[i])
            tangential = center - proj * pts[i]
            norm_t = np.linalg.norm(tangential)
            if norm_t > 1e-12:
                grad[i] += tangential / norm_t
    else:
        # Center is antipode of some point?
        # _covering_radius_and_facets returns center = -pts[i] in that case.
        # We need to know which i.
        # Let's find the point closest to -center (which is pts[i])
        # Actually, if center is antipode, then center = -p_k.
        # So p_k = -center.
        # Find index k such that pts[k] is closest to -center.
        dots = pts @ (-center)
        k = np.argmax(dots)
        # Move p_k away from center?
        # Distance is d(center, p_j) for all j.
        # The nearest point to center is p_k (which is -center).
        # To increase the minimum distance, we should move p_k away from center?
        # No, center is the worst-case point. We want to move the points to make center less "empty".
        # This means moving points TOWARDS center.
        # But p_k is at -center. Moving p_k towards center means moving it towards 0?
        # On the sphere, moving p_k towards center means moving it towards -p_k.
        # This increases the distance from p_k to center?
        # d(p_k, center) = d(p_k, -p_k) = pi.
        # If we move p_k slightly, the distance to -p_k changes.
        # Let p_k = (1,0,0). center = (-1,0,0).
        # Move p_k to (1, eps, 0). center is still (-1,0,0) initially?
        # The covering center moves too.
        # This gradient is tricky.
        # For simplicity, if the bottleneck is an antipode, it's hard to fix with local moves
        # unless we move the isolated point towards the cluster.
        # Let's push p_k towards the mean of other points.
        mask = np.ones(N, dtype=bool)
        mask[k] = False
        mean_other = np.sum(pts[mask], axis=0)
        mean_other = mean_other / np.linalg.norm(mean_other)
        
        proj = np.dot(mean_other, pts[k])
        tangential = mean_other - proj * pts[k]
        norm_t = np.linalg.norm(tangential)
        if norm_t > 1e-12:
            grad[k] += tangential / norm_t

    # Normalize gradient
    norms = np.linalg.norm(grad, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    grad = grad / norms
    
    new_pts = pts + step_size * grad
    return _to_sphere(new_pts)

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    # Start with Fibonacci spiral initial configuration
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _to_sphere(pts)
    
    best_pts = pts.copy()
    best_obj = _objective(pts)
    
    deadline = _DEADLINE
    remaining = deadline - time.time()
    
    # Phase 1: Lloyd's Algorithm
    if remaining > 10:
        pts, obj = _local_optimize_lloyd(pts, max_time=min(remaining - 10, 30))
        if obj < best_obj:
            best_obj = obj
            best_pts = pts.copy()
    
    # Phase 2: Gradient Refinement
    remaining = deadline - time.time()
    if remaining > 10:
        step_size = 0.05
        pts = best_pts.copy()
        current_obj = best_obj
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            new_pts = _gradient_step(pts, step_size)
            new_obj = _objective(new_pts)
            if new_obj < current_obj:
                pts = new_pts
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_pts = pts.copy()
            else:
                step_size *= 0.9
                if step_size < 0.001:
                    break
    
    # Phase 3: More Lloyd's with restarts if time permits
    remaining = deadline - time.time()
    if remaining > 20:
        np.random.seed(42)
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            perturbed = _to_sphere(best_pts + np.random.randn(N, 3) * 0.2)
            pts, obj = _local_optimize_lloyd(perturbed, max_time=min(remaining - 10, 15))
            if obj < best_obj:
                best_obj = obj
                best_pts = pts.copy()
    
    # Final Lloyd's
    remaining = deadline - time.time()
    if remaining > 5:
        pts, obj = _local_optimize_lloyd(best_pts, max_time=min(remaining - 5, 10))
        if obj < best_obj:
            best_obj = obj
            best_pts = pts.copy()
    
    best_pts = _to_sphere(best_pts)
    
    # Validate
    dots = best_pts @ best_pts.T
    np.fill_diagonal(dots, 0)
    if np.max(np.abs(dots)) > 0.999999:
        best_pts = _to_sphere(best_pts + np.random.randn(N, 3) * 0.01)
        
    return best_pts
# EVOLVE-BLOCK-END
