"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay

def _normalize(P):
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return P / norms

def _covering_radius(P):
    """
    Compute the covering radius in degrees for points P on the unit sphere.
    The covering radius is the maximum over the sphere of the angular distance
    to the nearest point in P. This equals the maximum circumradius of the
    spherical triangles in the spherical Delaunay triangulation, but a more
    robust approach is to find the maximum empty circle center.
    
    For points on a sphere, the covering radius can be found by:
    1. Computing the convex hull in 3D.
    2. For each face (triangle) of the hull, find the circumcenter of the 
       three points projected onto the sphere.
    3. The covering radius is the maximum angular distance from any such 
       circumcenter to its three vertices, but only if the circumcenter is 
       on the correct side (the face is "empty").
    
    Actually, a more direct approach: the covering radius is the maximum, over 
    all points on the sphere, of the min distance to P. This occurs at either:
    - A point in P (distance 0)
    - The circumcenter of a spherical triangle formed by 3 points in P, if that
      circumcenter is farther from all other points than from the 3 vertices.
    - The midpoint of a great circle arc between two points, if that arc is an
      edge of the Voronoi diagram.
    
    The most reliable computational method: use the fact that the covering radius
    equals the maximum, over all faces of the convex hull of P, of the angular
    radius of the circumscribed circle of that face (if the face's circumcenter
    is on the outward side). For a triangle on the sphere, the circumradius can
    be computed from the three side lengths.
    """
    # Ensure P is on unit sphere
    P = _normalize(P)
    
    # Compute convex hull
    try:
        hull = ConvexHull(P)
    except Exception:
        # Fallback: brute force on a fine grid
        # This is expensive but safe
        n_grid = 100
        theta = np.linspace(0, np.pi, n_grid)
        phi = np.linspace(0, 2*np.pi, n_grid)
        T, Ph = np.meshgrid(theta, phi)
        grid = np.column_stack([
            np.sin(T)*np.cos(Ph).ravel(),
            np.sin(T)*np.sin(Ph).ravel(),
            np.cos(T).ravel()
        ])
        # For each grid point, find min distance to P
        # This is O(n_grid^2 * N) which is 10000*50 = 500k, fine
        max_dist = 0.0
        for chunk_start in range(0, len(grid), 1000):
            chunk = grid[chunk_start:chunk_start+1000]
            # Compute distances
            # dist^2 = 2 - 2*dot
            dots = chunk @ P.T  # (chunk, N)
            dists = np.arccos(np.clip(1 - 0.5*(2 - 2*dots), -1, 1))
            # Wait, dot = cos(angle), so angle = arccos(dot)
            dists = np.arccos(np.clip(dots, -1, 1))
            min_dists = dists.min(axis=1)
            max_dist = max(max_dist, min_dists.max())
        return np.degrees(max_dist)
    
    # For each face of the convex hull, compute the circumradius
    # The covering radius is the max over all faces of the circumradius
    # of the spherical triangle, provided the circumcenter is "outside" 
    # (i.e., the face is part of the Voronoi diagram of the complement).
    
    max_cr = 0.0
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        p1, p2, p3 = P[simplex[0]], P[simplex[1]], P[simplex[2]]
        
        # Compute the circumcenter of the triangle on the sphere
        # The circumcenter is the point equidistant from p1, p2, p3 on the sphere.
        # It lies in the plane spanned by p1, p2, p3, on the unit sphere.
        # 
        # The circumcenter c satisfies:
        #   c . p1 = c . p2 = c . p3 = cos(r)
        #   ||c|| = 1
        #
        # This is a system: c = a*p1 + b*p2 + c_coeff*p3
        # with constraints. A simpler approach: the circumcenter is proportional to
        # the cross product of (p2-p1) and (p3-p1) ... no, that gives the normal.
        #
        # Actually, the circumcenter of three points on a sphere can be found by:
        # The center of the circumscribed circle of the spherical triangle.
        #
        # Let's use the formula: for a triangle with vertices p1, p2, p3 on unit sphere,
        # the circumradius r satisfies:
        #   sin(r) / sin(A) = sin(r) / sin(B) = sin(r) / sin(C) = R (circumradius in some sense)
        #
        # Actually, let's just compute it numerically. The circumcenter c is the 
        # intersection of the unit sphere with the plane that is the perpendicular 
        # bisector... no, on a sphere, the "perpendicular bisector" of a great circle 
        # arc is another great circle.
        #
        # Simpler: the circumcenter is the point on the unit sphere that maximizes the 
        # minimum distance to p1, p2, p3? No, that's what we're trying to find.
        #
        # Let's use the linear algebra approach. We want c such that:
        #   c · p1 = c · p2 = c · p3 = t
        #   c · c = 1
        #
        # From the first three: c · (p1 - p2) = 0, c · (p1 - p3) = 0
        # So c is perpendicular to (p1-p2) and (p1-p3), meaning c is parallel to 
        # (p1-p2) × (p1-p3).
        #
        # Let n = (p1-p2) × (p1-p3). Then c = ± n / ||n||.
        # We need to choose the sign such that c is on the correct side.
        
        v1 = p1 - p2
        v2 = p1 - p3
        n = np.cross(v1, v2)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        c = n / nn
        
        # Determine the correct sign: the circumcenter should be on the side of the 
        # face that is "away" from the origin... actually, for a convex hull face,
        # the outward normal points away from the centroid. The circumcenter of the 
        # spherical triangle corresponding to this face should be in the direction 
        # of the outward normal.
        
        # Check which side: compute the centroid of the face
        centroid = (p1 + p2 + p3) / 3.0
        # The outward normal of the face (for convex hull) points away from the origin
        # since the hull contains the origin (points on unit sphere, hull contains origin)
        # Actually, the face normal should point outward, i.e., same direction as centroid
        # (since centroid is inside the hull, the outward normal is opposite to centroid direction)
        # Wait: for a convex hull of points on the unit sphere, the origin is inside.
        # The outward normal of a face points away from the origin.
        # So the outward normal is in the direction of the face's centroid (which is 
        # on the same side as the face from the origin).
        
        # So c should be in the direction of the centroid (outward).
        if np.dot(c, centroid) < 0:
            c = -c
        
        # Now compute the angular distance from c to p1 (or p2 or p3)
        cos_r = np.dot(c, p1)
        cos_r = np.clip(cos_r, -1.0, 1.0)
        r = np.arccos(cos_r)
        
        # But we need to verify that c is actually the circumcenter, i.e., that 
        # c is equidistant from p1, p2, p3. Due to numerical issues, let's check.
        d1 = np.arccos(np.clip(np.dot(c, p1), -1, 1))
        d2 = np.arccos(np.clip(np.dot(c, p2), -1, 1))
        d3 = np.arccos(np.clip(np.dot(c, p3), -1, 1))
        
        # If they're not approximately equal, this might not be a valid circumcenter
        # (could be the "antipodal" circumcenter or numerical error)
        if max(d1, d2, d3) - min(d1, d2, d3) > 0.1:
            # Try the other sign
            c = -c
            d1 = np.arccos(np.clip(np.dot(c, p1), -1, 1))
            d2 = np.arccos(np.clip(np.dot(c, p2), -1, 1))
            d3 = np.arccos(np.clip(np.dot(c, p3), -1, 1))
            if max(d1, d2, d3) - min(d1, d2, d3) > 0.1:
                continue
        
        r = max(d1, d2, d3)
        
        # We also need to check that this circumcenter is not closer to some other 
        # point in P. If it is, then this triangle's circumcircle is not empty, 
        # and the actual covering radius contribution from this region is smaller.
        # However, for the covering radius, we need the MAX over all points on the 
        # sphere. The point c is a candidate local maximum of the min-distance function.
        # If c is closer to some other point, then c is not a local maximum, and the 
        # actual local maximum in this region would be at a different location (likely 
        # on an edge of the Voronoi cell).
        
        # For now, let's just take the max circumradius over all faces. This gives 
        # an upper bound. To get the exact covering radius, we'd need to check 
        # emptiness. Let's do a quick check.
        
        # Check if c is closer to any other point
        dots_all = P @ c
        min_dist_to_other = np.arccos(np.clip(dots_all.max(), -1, 1))
        
        # The min distance from c to any point in P
        min_d = min(d1, d2, d3)
        
        # If c is a valid Voronoi vertex (equidistant to 3 points and no closer to others)
        # then the covering radius candidate is min_d (which equals d1=d2=d3).
        # If c is closer to some other point, then the covering radius in this region 
        # is determined by the edge between c's nearest point and the second nearest.
        
        # For a robust estimate, let's just use the circumradius if the face is "empty"
        # in the sense that no other point is closer to c than the circumradius.
        
        if min_dist_to_other >= min_d - 1e-6:
            max_cr = max(max_cr, min_d)
        else:
            # The circumcenter is not a Voronoi vertex. The covering radius in this 
            # region is determined by an edge. Let's find the two closest points to c.
            dists = np.arccos(np.clip(dots_all, -1, 1))
            idx_sorted = np.argsort(dists)
            # The covering radius at the midpoint of the great circle arc between 
            # the two closest points (if that midpoint is farther from all others)
            # is half the angular distance between them.
            # Actually, the local maximum on the edge between point i and point j 
            # (where i and j are the two closest to c) is at the midpoint of the 
            # great circle arc, and the distance is half the arc length.
            i, j = idx_sorted[0], idx_sorted[1]
            arc_len = dists[i]  # Wait, dists[i] is the distance from c to point i, 
                                # not the arc length between i and j.
            # Let's compute the arc length between i and j
            cos_ij = np.dot(P[i], P[j])
            arc_ij = np.arccos(np.clip(cos_ij, -1, 1))
            # The midpoint of the arc is at distance arc_ij/2 from both i and j.
            # But we need to check if this midpoint is farther from all other points.
            # For simplicity, let's just use arc_ij/2 as a candidate.
            max_cr = max(max_cr, arc_ij / 2.0)
    
    return np.degrees(max_cr)


def _initial_fibonacci():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _gradient_step(P, lr=0.01, iters=1000):
    """
    Gradient descent to minimize the covering radius.
    The objective is max over y of min_i angle(y, P_i).
    This is nonsmooth, so we use a smooth surrogate: 
    f(P) = log-sum-exp of the "active" constraints.
    
    Alternatively, we can use the fact that the covering radius is determined by 
    the Voronoi vertices (circumcenters of triangular faces). We can move points 
    to reduce the maximum circumradius.
    
    A simpler approach: for each point P_i, compute the gradient of the 
    min-distance function at the current covering radius location, and move 
    P_i in the direction that reduces it.
    
    Actually, let's use a simpler heuristic: for each point, move it away from 
    the nearest Voronoi vertex that determines the covering radius.
    """
    P = _normalize(P.copy())
    
    for it in range(iters):
        # Compute covering radius and identify the "worst" face
        # For simplicity, let's just do a random perturbation with acceptance
        # based on covering radius improvement. This is essentially simulated 
        # annealing or random hill climbing.
        
        # Compute current covering radius
        cr = _covering_radius(P)
        
        # Pick a random point to move
        idx = np.random.randint(N)
        
        # Generate a random direction on the tangent plane at P[idx]
        v = np.random.randn(3)
        v -= np.dot(v, P[idx]) * P[idx]
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            continue
        v = v / vn
        
        # Move the point
        new_P = P.copy()
        new_P[idx] = P[idx] + lr * v
        new_P[idx] = new_P[idx] / np.linalg.norm(new_P[idx])
        
        # Compute new covering radius
        new_cr = _covering_radius(new_P)
        
        # Accept if improved
        if new_cr < cr:
            P = new_P
    
    return P


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with Fibonacci spiral
    P = _initial_fibonacci()
    best_P = P.copy()
    best_cr = _covering_radius(best_P)
    
    # Try a few random restarts with gradient-based refinement
    np.random.seed(42)
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            P = _initial_fibonacci()
        else:
            # Random initialization
            P = np.random.randn(N, 3)
            P = _normalize(P)
        
        # Gradient-based refinement
        P = _gradient_step(P, lr=0.05, iters=200)
        
        cr = _covering_radius(P)
        if cr < best_cr:
            best_cr = cr
            best_P = P.copy()
    
    # Final polish: finer steps
    if time.time() < deadline - 10.0:
        P = _gradient_step(best_P, lr=0.01, iters=500)
        cr = _covering_radius(P)
        if cr < best_cr:
            best_cr = cr
            best_P = P.copy()
    
    return _normalize(best_P)
# EVOLVE-BLOCK-END
