"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _covering_radius_deg(pts):
    """Compute covering radius in degrees via Delaunay triangulation on sphere."""
    from scipy.spatial import ConvexHull
    # Compute convex hull to find facets
    hull = ConvexHull(pts)
    facets = hull.simplices
    max_circum = 0.0
    for tri in facets:
        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        # Check if this facet's circumcenter is valid (cap is empty)
        # Compute circumcenter of triangle on sphere
        # The circumcenter of three points on a unit sphere
        # Cross product gives normal to the plane
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-10:
            continue
        n = n / nn
        # The circumcenter is along n from origin
        # Check if all other points are on the same side as n (empty cap)
        dots = pts @ n
        # Points on the "wrong" side would have dot < 0 (opposite to n)
        # But we need to check if any point is further from a,b,c than the circumradius
        # Actually, for a valid Delaunay facet, the cap is empty by construction
        # The circumradius of the spherical triangle
        # Use formula: for three points on unit sphere, the circumradius R satisfies
        # cos(R) = (a·b + b·c + c·a - 1) / (2 * (a·(b×c)))... let me use a simpler approach
        # The angular circumradius can be found from the triangle sides
        ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
        bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
        ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
        # Spherical circumradius formula
        # Using the formula: R = arccos( (cos(a)cos(b) + cos(c)) / (2*sin(a)*sin(b)) )
        # where a,b,c are the angles at the vertices... this is complex
        # Alternative: the circumcenter is at distance d from each vertex
        # On the sphere, if circumcenter is at point C, then angular dist from C to a = R
        # C is along the normal n (or -n). Let's compute R from the triangle.
        # For a spherical triangle with sides a,b,c (in radians), the circumradius R satisfies:
        # sin(R) = sin(a)sin(b)sin(c) / (2 * area_factor)... 
        # Let me just compute the circumcenter numerically
        # The circumcenter C satisfies |C-a|^2 = |C-b|^2 = |C-c|^2 and |C|=1
        # This means C·(a-b) = (|a|^2 - |b|^2)/2 = 0, etc.
        # So C is perpendicular to (a-b) and (b-c) in the ambient space, i.e., C is parallel to a×(b×c) type thing
        # Actually C is parallel to n = (b-a)×(c-a)
        # Let C = t*n. Then |C|=1 means t = 1/nn
        # But we also need C to be on the correct side (the cap side)
        # Check: C should have positive dot with the triangle centroid
        centroid = (a + b + c) / 3
        if np.dot(n, centroid) < 0:
            n = -n
        C = n  # since nn was already normalized
        # Now R = arccos(C·a)
        cos_R = np.clip(np.dot(C, a), -1, 1)
        R = np.arccos(cos_R)
        max_circum = max(max_circum, R)
    return np.degrees(max_circum)


def _smooth_covering_radius(pts, beta=100.0):
    """Smooth surrogate for covering radius using log-sum-exp over antipodal points."""
    # For each point, compute its distance to all others
    # The covering radius is related to the largest empty cap
    # A good smooth approximation: maximize over y of min_i d(y, x_i)
    # We can use the fact that the worst point is either at a vertex of the Voronoi diagram
    # or at the center of a facet
    # For smooth optimization, we use a soft version
    # Compute pairwise angular distances
    dots = np.clip(pts @ pts.T, -1, 1)
    dists = np.arccos(dots)
    # For each point i, the "local covering radius" is half the max distance to nearest neighbor
    # Actually, we want to minimize the max over all sphere points of min distance
    # A proxy: for each point, compute its distance to its 2 nearest neighbors
    # The covering radius is approximately the max over i of 0.5 * (dist to 2nd nearest)
    # But this is only approximate. Let's use a different approach.
    
    # Better: use the convex hull approach but make it smooth
    # For now, use a gradient-free approach with the actual covering radius
    return _covering_radius_deg(pts)


def _grad_covering_radius(pts):
    """Compute gradient of a smooth surrogate for covering radius."""
    # Use the idea: for each pair of points, the midpoint's distance to both
    # The covering radius is bounded below by half the minimum edge length
    # and above by the circumradius of the largest facet
    
    # Simple gradient: push points apart based on their nearest neighbors
    dots = np.clip(pts @ pts.T, -1, 1)
    dists = np.arccos(dots)
    np.fill_diagonal(dists, np.inf)
    
    # For each point, find its 2 nearest neighbors
    idx_sorted = np.argsort(dists, axis=1)
    nn1 = idx_sorted[:, 0]
    nn2 = idx_sorted[:, 1]
    
    # The "local covering radius" for point i is approximately 0.5 * dist(i, nn2(i))
    # but this is a lower bound on the actual covering radius
    # Gradient: move point i away from its nearest neighbor
    grad = np.zeros_like(pts)
    for i in range(N):
        j = nn1[i]
        # Move i away from j
        # The gradient of dist(i,j) w.r.t. i is -j/sqrt(1-(i·j)^2) projected to tangent
        dot_ij = np.dot(pts[i], pts[j])
        dot_ij = np.clip(dot_ij, -1, 1)
        if abs(dot_ij) < 1 - 1e-10:
            # Tangent direction away from j
            tang = pts[i] - dot_ij * pts[j]
            tn = np.linalg.norm(tang)
            if tn > 1e-10:
                tang = tang / tn
                grad[i] += tang / (1 - dot_ij)  # weight by inverse distance
    return grad


def construct():
    t0 = time.time()
    deadline = 140.0  # safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_score = _covering_radius_deg(pts)
    
    # Gradient descent with adaptive step size
    lr = 0.01
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        grad = _grad_covering_radius(pts)
        # Normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-10:
            break
        grad = grad / gnorm
        
        # Update
        new_pts = pts + lr * grad
        new_pts = _normalize(new_pts)
        
        new_score = _covering_radius_deg(new_pts)
        if new_score < best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts
            # Decrease learning rate slightly
            lr *= 0.99
        else:
            # Try smaller step
            lr *= 0.5
            if lr < 1e-6:
                break
    
    return best_pts
# EVOLVE-BLOCK-END
