"""Improved program for covering the sphere with 50 equal caps."""
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.optimize import minimize

N = 50
SAMPLES_COUNT = 2000

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius_exact(pts):
    """
    Compute the exact covering radius in degrees using Convex Hull facets.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    max_radius_deg = 0.0
    for simplex in hull.simplices:
        p0 = pts[simplex[0]]
        p1 = pts[simplex[1]]
        p2 = pts[simplex[2]]
        
        a = p1 - p0
        b = p2 - p0
        aa = np.dot(a, a)
        bb = np.dot(b, b)
        ab = np.dot(a, b)
        
        denom = aa * bb - ab * ab
        if abs(denom) < 1e-15:
            continue
            
        s = (aa * bb - ab * ab) / (2.0 * denom)
        t = (aa * ab - aa * aa) / (2.0 * denom)
        
        circumcenter_2d = p0 + s * a + t * b
        R = np.linalg.norm(circumcenter_2d - p0)
        C_norm = np.linalg.norm(circumcenter_2d)
        
        if C_norm < 1e-15:
            theta = np.pi / 2.0
        else:
            cos_theta = (C_norm * C_norm + 1.0 - R * R) / (2.0 * C_norm)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
        
        radius_deg = np.degrees(theta)
        if radius_deg > max_radius_deg:
            max_radius_deg = radius_deg
            
    return max_radius_deg

def _generate_samples(n_samples, seed=0):
    """Generate uniform random points on the sphere for fast approximation."""
    rng = np.random.RandomState(seed)
    # Use normal distribution and normalize
    pts = rng.randn(n_samples, 3)
    return _normalize(pts)

def _covering_radius_fast(pts, samples):
    """
    Approximate covering radius by checking distance to a fixed set of sample points.
    """
    pts = _normalize(pts)
    tree = cKDTree(pts)
    # Euclidean distance d is related to angular distance theta by d = 2*sin(theta/2)
    # Minimizing d is equivalent to minimizing theta.
    # d_max = max(min_dist_to_sample)
    # theta_max = 2 * arcsin(d_max / 2)
    dists, _ = tree.query(samples)
    d_max = np.max(dists)
    # Clamp d_max to [0, 2] for numerical stability
    d_max = np.clip(d_max, 0.0, 2.0)
    theta_max = 2.0 * np.arcsin(d_max / 2.0)
    return np.degrees(theta_max)

def _generate_icosahedron_start():
    """
    Generate a starting configuration based on icosahedron vertices 
    and points distributed in the faces.
    """
    # Icosahedron vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ], dtype=float)
    verts = _normalize(verts)
    
    # We have 12 vertices. We need 50 points.
    # 50 - 12 = 38 points to add.
    # There are 20 faces. 38 / 20 = 1.9. 
    # We can place 1 or 2 points per face.
    # Let's use a simpler robust start: Fibonacci is often better for optimization start than structured ones if the structure is wrong.
    # But let's try a "Geodesic Dome" like start:
    # Subdivide the icosahedron faces.
    
    # For simplicity and robustness in this new program, we will stick to 
    # Fibonacci and Random starts, but the KEY is the Fast Objective.
    # The icosahedron start is complex to generate correctly for 50 pts without 
    # pre-computing the subdivision. 
    # Let's just return the icosahedron vertices + random fills for the rest? 
    # No, that might be poor. 
    # Let's just use Fibonacci and Random in construct().
    return None 

def construct():
    """
    Multi-start optimization using a fast approximate objective function,
    followed by exact polishing.
    """
    samples = _generate_samples(SAMPLES_COUNT)
    best_pts = None
    best_radius = np.inf
    
    # Initial configurations
    inits = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(_normalize(fib))
    
    # 2. Random starts
    rng = np.random.RandomState(42)
    for _ in range(5):
        pts = rng.randn(N, 3)
        pts = _normalize(pts)
        inits.append(pts)

    # 3. "Clustered" start: 5 clusters of 10 points? 
    # Sometimes helps escape local minima.
    # Let's stick to the standard ones for now to ensure reliability.

    for init in inits:
        try:
            # Phase 1: Fast Optimization
            # Use Powell or Nelder-Mead with the fast objective
            def objective_fast(params):
                pts = _normalize(params.reshape(N, 3))
                return _covering_radius_fast(pts, samples)

            result_fast = minimize(
                objective_fast,
                init.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            pts_candidate = _normalize(result_fast.x.reshape(N, 3))
            
            # Phase 2: Exact Polishing
            # Use the exact objective for a few iterations to refine
            def objective_exact(params):
                pts = _normalize(params.reshape(N, 3))
                return _covering_radius_exact(pts)

            result_exact = minimize(
                objective_exact,
                pts_candidate.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            pts_final = _normalize(result_exact.x.reshape(N, 3))
            radius = _covering_radius_exact(pts_final)
            
            if radius < best_radius:
                best_radius = radius
                best_pts = pts_final
                
        except Exception:
            continue

    if best_pts is None:
        best_pts = _normalize(inits[0])
        
    return best_pts
