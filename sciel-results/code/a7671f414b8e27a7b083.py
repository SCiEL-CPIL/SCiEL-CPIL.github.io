"""
Geometric Facet Polishing for Covering the Sphere with 50 Equal Caps.

Approach:
1. Initialize points using diverse methods (Fibonacci, Random, Icosahedron).
2. Iteratively identify the convex hull facet with the largest circumradius.
3. Perform local optimization on the 3 vertices of that facet to minimize the
   maximum circumradius of that facet and its neighbors.
4. Repeat until convergence or time limit.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _facet_circumradius(p0, p1, p2):
    """
    Calculate the angular circumradius of a spherical triangle defined by 3 unit vectors.
    Returns radius in radians.
    """
    # Vectors from p0
    a = p1 - p0
    b = p2 - p0
    
    aa = np.dot(a, a)
    bb = np.dot(b, b)
    ab = np.dot(a, b)
    
    denom = aa * bb - ab * ab
    if abs(denom) < 1e-15:
        # Degenerate triangle
        # Return max distance to p0 or pi/2? 
        # If collinear, the circumcenter is at infinity or undefined.
        # For covering radius, we can treat it as pi/2 or check neighbors.
        # Usually, degeneracy implies points are very close or antipodal.
        # Let's return a large value to flag it, or 0 if points are identical.
        if np.linalg.norm(a) < 1e-10 or np.linalg.norm(b) < 1e-10:
            return 0.0
        return np.pi / 2.0
        
    # Barycentric coordinates of circumcenter relative to p0
    # Using the formula for circumcenter in 2D triangle embedded in 3D
    # s, t such that C = p0 + s*a + t*b
    s = (aa * bb - ab * ab) / (2.0 * denom)
    t = (aa * ab - aa * aa) / (2.0 * denom)
    
    circumcenter = p0 + s * a + t * b
    R_chord = np.linalg.norm(circumcenter - p0)
    C_norm = np.linalg.norm(circumcenter)
    
    if C_norm < 1e-15:
        return np.pi / 2.0
        
    # Relationship between chord radius R, sphere radius 1, and angular radius theta
    # R = C_norm * sin(theta) ? No.
    # The circumcenter C is at distance C_norm from origin.
    # The points are on sphere of radius 1.
    # Distance from C to point p0 is R_chord.
    # Triangle O-C-p0 is right-angled at ? No.
    # C is the circumcenter of the triangle on the sphere.
    # The plane of the triangle is at distance d from origin?
    # Actually, simpler:
    # cos(theta) = (C_norm^2 + 1 - R_chord^2) / (2 * C_norm * 1)
    cos_theta = (C_norm * C_norm + 1.0 - R_chord * R_chord) / (2.0 * C_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    return theta

def _get_hull_and_radii(pts):
    """
    Compute convex hull and circumradius for each facet.
    Returns:
        hull: ConvexHull object
        radii: array of circumradii (radians) for each facet
        simplices: array of vertex indices for each facet
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, None
        
    radii = np.zeros(len(hull.simplices))
    for i, simplex in enumerate(hull.simplices):
        p0 = pts[simplex[0]]
        p1 = pts[simplex[1]]
        p2 = pts[simplex[2]]
        radii[i] = _facet_circumradius(p0, p1, p2)
        
    return hull, radii, hull.simplices

def _local_optimize_worst_facet(pts, worst_idx, max_iter=100):
    """
    Optimize the 3 vertices of the worst facet to minimize local covering radius.
    """
    hull, radii, simplices = _get_hull_and_radii(pts)
    if hull is None:
        return pts
        
    # Find worst facet
    max_rad_idx = np.argmax(radii)
    worst_simplex = simplices[max_rad_idx]
    v_indices = worst_simplex
    
    # Get neighbors of these vertices
    # A facet is a neighbor if it shares at least one vertex? 
    # Actually, to ensure we don't break the hull or increase radius elsewhere,
    # we should check facets that share edges or vertices.
    # Sharing vertices is sufficient for local consistency.
    neighbor_facets = set()
    for f_idx, simplex in enumerate(simplices):
        for v in v_indices:
            if v in simplex:
                neighbor_facets.add(f_idx)
                break
                
    neighbor_list = list(neighbor_facets)
    
    # Objective function for the 3 vertices
    # We parameterize the 3 vertices as unit vectors.
    # We use a flat vector of 9 parameters.
    
    def local_objective(params):
        new_pts = pts.copy()
        # Reshape params to 3x3
        new_verts = params.reshape(3, 3)
        # Normalize
        new_verts = _normalize(new_verts)
        
        # Update points
        new_pts[v_indices[0]] = new_verts[0]
        new_pts[v_indices[1]] = new_verts[1]
        new_pts[v_indices[2]] = new_verts[2]
        
        # Recompute hull and radii for neighbors
        try:
            hull_new = ConvexHull(new_pts)
        except Exception:
            return 10.0 # Penalize invalid hull
            
        max_local_rad = 0.0
        # We only need to check neighbors, but to be safe, we can check all 
        # or just the ones connected to the moved vertices.
        # Checking all is O(N) which is fine for N=50.
        for f_idx in neighbor_list:
            # Check if this facet still exists in the new hull?
            # The hull might change topology. 
            # Simplest: just compute max radius of ALL facets in new hull.
            # This is safer but slightly more expensive.
            pass
            
        # For simplicity and robustness, let's just compute the max radius of the whole new configuration.
        # It's only 50 points, so ConvexHull is fast.
        hull_new, radii_new, _ = _get_hull_and_radii(new_pts)
        if radii_new is None:
            return 10.0
        return np.max(radii_new)
        
    # Initial parameters
    x0 = np.zeros(9)
    x0[0:3] = pts[v_indices[0]]
    x0[3:6] = pts[v_indices[1]]
    x0[6:9] = pts[v_indices[2]]
    
    try:
        res = minimize(local_objective, x0, method='Nelder-Mead', 
                       options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if res.success and res.fun < np.max(radii):
            new_pts = pts.copy()
            new_verts = _normalize(res.x.reshape(3, 3))
            new_pts[v_indices[0]] = new_verts[0]
            new_pts[v_indices[1]] = new_verts[1]
            new_pts[v_indices[2]] = new_verts[2]
            return new_pts
    except Exception:
        pass
        
    return pts

def _fibonacci_sphere(n, offset=0.5):
    """Generate points on a sphere using Fibonacci spiral."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return _normalize(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

def construct():
    """
    Geometric Facet Polishing for minimizing covering radius.
    """
    best_pts = None
    best_radius = np.inf
    start_time = time.time()
    
    # Initial configurations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(_fibonacci_sphere(N))
    
    # 2. Random starts
    rng = np.random.RandomState(42)
    for _ in range(5):
        pts = rng.randn(N, 3)
        inits.append(_normalize(pts))
        
    # 3. Icosahedron based
    phi_golden = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [-1,  phi_golden, 0], [ 1,  phi_golden, 0], [-1, -phi_golden, 0], [ 1, -phi_golden, 0],
        [0, -1,  phi_golden], [0,  1,  phi_golden], [0, -1, -phi_golden], [0,  1, -phi_golden],
        [ phi_golden, 0, -1], [ phi_golden, 0,  1], [-phi_golden, 0, -1], [-phi_golden, 0,  1]
    ], dtype=float)
    ico_verts = _normalize(ico_verts)
    remaining = N - len(ico_verts)
    extra = rng.randn(remaining, 3)
    extra = _normalize(extra)
    inits.append(_normalize(np.vstack([ico_verts, extra])))
    
    for init_idx, init in enumerate(inits):
        if time.time() - start_time > 100: # Leave 20s buffer
            break
            
        pts = init.copy()
        current_radius = np.inf
        no_improve_count = 0
        
        # Polishing loop
        for step in range(200):
            if time.time() - start_time > 110:
                break
                
            hull, radii, simplices = _get_hull_and_radii(pts)
            if radii is None:
                break
                
            current_radius = np.max(radii)
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # If we've stuck, perturb
            if no_improve_count > 20:
                pts = _normalize(pts + 0.05 * rng.randn(*pts.shape))
                no_improve_count = 0
                continue
                
            # Optimize worst facet
            pts_new = _local_optimize_worst_facet(pts)
            
            # Check if it improved
            hull_new, radii_new, _ = _get_hull_and_radii(pts_new)
            if radii_new is not None:
                new_radius = np.max(radii_new)
                if new_radius < current_radius:
                    pts = pts_new
                else:
                    # If local opt didn't help, maybe try a small random step on the worst vertices
                    worst_idx = np.argmax(radii)
                    v_idx = simplices[worst_idx]
                    for v in v_idx:
                        pts[v] = _normalize(pts[v] + 0.01 * rng.randn(3))
            else:
                break
                
        # End of polishing for this init
        
    if best_pts is None:
        best_pts = inits[0]
        
    return best_pts
