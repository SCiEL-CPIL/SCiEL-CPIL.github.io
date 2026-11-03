"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # safety margin before 150s kill

def _to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius(pts):
    """
    Compute the covering radius (in radians) of a set of points on the sphere.
    """
    pts = _to_sphere(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return _covering_radius_fallback(pts)
    
    max_radius = 0.0
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Circumcenter of the triangle on the sphere
        # x is parallel to (a-b) x (a-c)
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            continue
        
        x = cross / norm_cross
        
        # The circumcenter is the point on the unit sphere equidistant from a, b, c.
        # There are two candidates: x and -x.
        # We want the one that is "inside" the cap defined by the triangle, 
        # which corresponds to the smaller angular distance to the vertices 
        # if the triangle is acute, but generally we check both.
        # Actually, for the covering radius, we are looking for the point 
        # farthest from all points. For a facet, the "worst" point is the 
        # circumcenter of that facet. 
        # The angular distance from the circumcenter to the vertices is the 
        # circumradius of the facet.
        
        dot_a = np.clip(np.dot(x, a), -1.0, 1.0)
        r_pos = np.arccos(dot_a)
        
        dot_a_neg = np.clip(np.dot(-x, a), -1.0, 1.0)
        r_neg = np.arccos(dot_a_neg)
        
        # The circumradius of the facet is the distance from the circumcenter 
        # to the vertices. We take the one that corresponds to the center of 
        # the empty cap. 
        # For a convex hull facet, the origin is on one side. The circumcenter 
        # of the spherical triangle is the one on the same side as the origin 
        # relative to the plane of the triangle? No, the circumcenter of the 
        # spherical triangle is the pole of the great circle through the 
        # vertices? No.
        
        # Let's just take the max of the two possible circumradii. 
        # Actually, for a triangle on the sphere, the circumradius R is 
        # determined by the side lengths. 
        # The point x (normalized cross product) is the pole of the plane 
        # containing a, b, c. The angular distance from x to a is the 
        # same as from x to b and x to c.
        # This distance is the circumradius of the spherical triangle 
        # IF the triangle is acute. If obtuse, the circumcenter is outside.
        # In any case, the covering radius candidate for this facet is 
        # the distance from the circumcenter to the vertices.
        
        # Which of x or -x is the correct circumcenter?
        # The circumcenter should be such that the triangle is contained 
        # in the cap centered at the circumcenter with radius R.
        # For a facet of the convex hull, the "empty" region is on the 
        # side of the facet away from the origin. 
        # The normal vector of the facet (outward) is n = (b-a) x (c-a) / |...|
        # The circumcenter should be in the direction of n?
        
        # Let's check the dot product of x with the outward normal.
        n = np.cross(b - a, c - a)
        n = n / np.linalg.norm(n)
        # Ensure n points away from origin
        if np.dot(n, a) < 0:
            n = -n
        
        # If x is in the same direction as n, it's the circumcenter on the 
        # "outside". Otherwise -x.
        # Actually, the circumcenter of the spherical triangle is the point 
        # on the sphere equidistant from the vertices. 
        # Both x and -x are equidistant. 
        # The one that is the center of the minimal cap containing the 
        # triangle is the one closer to the triangle.
        # For a facet, the "hole" is on the side of the outward normal.
        # So the circumcenter of the hole is the one in the direction of n.
        
        if np.dot(x, n) > 0:
            candidate_r = r_pos
        else:
            candidate_r = r_neg
            
        max_radius = max(max_radius, candidate_r)
        
    # Also check antipodes of points
    for i in range(len(pts)):
        neg_pt = -pts[i]
        dists = np.arccos(np.clip(pts @ neg_pt, -1.0, 1.0))
        min_dist = np.min(dists)
        max_radius = max(max_radius, min_dist)
        
    return max_radius

def _covering_radius_fallback(pts):
    """Fallback covering radius computation via dense sampling."""
    n_theta = 200
    n_phi = 400
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    max_radius = 0.0
    for theta in thetas:
        st = np.sin(theta)
        ct = np.cos(theta)
        for phi in phis:
            y = np.array([st*np.cos(phi), st*np.sin(phi), ct])
            dists = np.arccos(np.clip(pts @ y, -1.0, 1.0))
            min_dist = np.min(dists)
            if min_dist > max_radius:
                max_radius = min_dist
    return max_radius

def _objective(pts):
    """Compute the objective (covering radius in radians)."""
    return _covering_radius(pts)

def _spherical_lloyd_iteration(pts):
    """
    Perform one iteration of Spherical Lloyd's algorithm.
    Moves each point to the spherical centroid of its Voronoi cell.
    """
    pts = _to_sphere(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return pts, False
    
    # Build Delaunay triangulation from ConvexHull
    # Each simplex in hull.simplices is a triangle (i, j, k)
    # The Delaunay triangulation is the dual of the Voronoi diagram.
    # For each point i, we need to find all triangles (facets) that contain i.
    # The Voronoi cell of point i is the set of points on the sphere closer to i 
    # than to any other point.
    # The vertices of the Voronoi cell are the circumcenters of the triangles 
    # in the Delaunay triangulation that contain i.
    
    # For each point, collect the circumcenters of the adjacent triangles.
    # The spherical centroid is the normalized sum of these circumcenters.
    
    # Map: point index -> list of circumcenters
    circumcenters = [[] for _ in range(N)]
    
    for simplex in hull.simplices:
        i, j, k = simplex
        a = pts[i]
        b = pts[j]
        c = pts[k]
        
        # Compute circumcenter
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            continue
            
        x = cross / norm_cross
        
        # Determine correct direction (outward normal side)
        n = np.cross(b - a, c - a)
        n = n / np.linalg.norm(n)
        if np.dot(n, a) < 0:
            n = -n
            
        if np.dot(x, n) > 0:
            cc = x
        else:
            cc = -x
            
        circumcenters[i].append(cc)
        circumcenters[j].append(cc)
        circumcenters[k].append(cc)
        
    new_pts = np.zeros_like(pts)
    valid = True
    
    for i in range(N):
        if not circumcenters[i]:
            # Point is not part of any triangle (isolated or degenerate)
            new_pts[i] = pts[i]
            continue
            
        # Spherical centroid: normalize the sum of the circumcenters
        # Weighted by the area of the corresponding triangle? 
        # For simplicity, uniform weight is often sufficient for Lloyd's.
        centroid = np.sum(circumcenters[i], axis=0)
        norm_c = np.linalg.norm(centroid)
        
        if norm_c < 1e-12:
            new_pts[i] = pts[i]
        else:
            new_pts[i] = centroid / norm_c
            
    new_pts = _to_sphere(new_pts)
    return new_pts, True

def _local_optimize(pts, max_time=None):
    """
    Local optimization using Spherical Lloyd's algorithm.
    """
    if max_time is None:
        max_time = 10.0
    
    pts = _to_sphere(pts.copy())
    current_obj = _objective(pts)
    best_pts = pts.copy()
    best_obj = current_obj
    
    start = time.time()
    
    while time.time() - start < max_time:
        new_pts, valid = _spherical_lloyd_iteration(pts)
        if not valid:
            break
            
        new_obj = _objective(new_pts)
        
        if new_obj < current_obj:
            pts = new_pts
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_pts = pts.copy()
        else:
            # No improvement, stop
            break
            
    return best_pts, best_obj

def _simulated_annealing(pts, max_time):
    """
    Simulated annealing to find a good configuration.
    """
    pts = _to_sphere(pts.copy())
    current_obj = _objective(pts)
    best_pts = pts.copy()
    best_obj = current_obj
    
    start = time.time()
    temperature = 0.1
    cooling = 0.999
    step_size = 0.1
    
    while time.time() < max_time:
        # Propose a random perturbation
        noise = np.random.randn(N, 3) * step_size
        new_pts = _to_sphere(pts + noise)
        
        new_obj = _objective(new_pts)
        
        delta = new_obj - current_obj
        if delta < 0 or np.random.random() < np.exp(-delta / temperature):
            pts = new_pts
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_pts = pts.copy()
        
        temperature *= cooling
        if temperature < 0.001:
            temperature = 0.001
            step_size = 0.01
            
    return best_pts, best_obj

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    # Start with a good initial configuration (Fibonacci spiral)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _to_sphere(pts)
    
    best_pts = pts.copy()
    best_obj = _objective(pts)
    
    # Local optimization using Spherical Lloyd's
    deadline = _START_TIME + 145.0
    remaining = deadline - time.time()
    
    if remaining > 5:
        pts, obj = _local_optimize(pts, max_time=min(remaining - 5, 30))
        if obj < best_obj:
            best_obj = obj
            best_pts = pts.copy()
    
    # Simulated annealing with multiple restarts
    remaining = deadline - time.time()
    if remaining > 10:
        np.random.seed(42)
        for restart in range(5):
            if time.time() > deadline - 10:
                break
                
            # Create a perturbed version of the best solution
            perturbed = _to_sphere(best_pts + np.random.randn(N, 3) * 0.5)
            pts, obj = _simulated_annealing(perturbed, max_time=min(remaining - 10, 20))
            
            if obj < best_obj:
                best_obj = obj
                best_pts = pts.copy()
                
            # Also try local optimization (Lloyd's) from the SA result
            remaining = deadline - time.time()
            if remaining > 10:
                pts, obj = _local_optimize(pts, max_time=min(remaining - 10, 15))
                if obj < best_obj:
                    best_obj = obj
                    best_pts = pts.copy()
                    
    # Final local optimization
    remaining = deadline - time.time()
    if remaining > 5:
        pts, obj = _local_optimize(best_pts, max_time=min(remaining - 5, 20))
        if obj < best_obj:
            best_obj = obj
            best_pts = pts.copy()
            
    # Ensure all points are on the unit sphere
    best_pts = _to_sphere(best_pts)
    
    # Validate: check for coincident points and hemisphere issues
    # Check for coincident points
    dots = best_pts @ best_pts.T
    np.fill_diagonal(dots, 0)
    if np.max(np.abs(dots)) > 0.999999:
        # Points are too close, add small perturbation
        best_pts = _to_sphere(best_pts + np.random.randn(N, 3) * 0.01)
        
    # Check if all points are in a closed hemisphere
    # A set of points is in a closed hemisphere if there exists a vector v
    # such that v·x_i >= 0 for all i
    # This is equivalent to the origin being in the convex hull of the points
    # (by Gordan's theorem)
    
    # Check if the origin is in the convex hull
    # Use a linear programming approach or just check the convex hull
    try:
        hull = ConvexHull(best_pts)
        # Check if the origin is inside the convex hull
        # The origin is inside if all facets have the origin on the same side
        for simplex in hull.simplices:
            a = best_pts[simplex[0]]
            b = best_pts[simplex[1]]
            c = best_pts[simplex[2]]
            normal = np.cross(b - a, c - a)
            normal = normal / np.linalg.norm(normal)
            # Check if origin is on the negative side of the facet
            if np.dot(normal, -a) > 0:
                # Origin is outside the convex hull, so points are in a hemisphere
                # Add a point at the antipode to fix this
                pass
    except Exception:
        pass
        
    return best_pts
# EVOLVE-BLOCK-END
