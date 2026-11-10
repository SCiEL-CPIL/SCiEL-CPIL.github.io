import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _normalize(v):
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return v
        return v / norm
    else:
        norms = np.linalg.norm(v, axis=1)
        norms[norms < 1e-12] = 1.0
        return v / norms[:, np.newaxis]

def _covering_radius(pts):
    """
    Compute the exact covering radius using the convex hull facets.
    """
    pts = _normalize(pts)
    
    # If points are not in general position or hull fails, fallback to sampling
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback to sampling if hull fails
        n_samples = 10000
        i = np.arange(n_samples) + 0.5
        z = 1.0 - 2.0 * i / n_samples
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        dots = samples @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        max_dots = np.max(dots, axis=1)
        angles = np.arccos(max_dots)
        return np.max(angles) * 180.0 / np.pi

    max_radius = 0.0
    
    # Iterate over facets
    for simplex in hull.simplices:
        v1 = pts[simplex[0]]
        v2 = pts[simplex[1]]
        v3 = pts[simplex[2]]
        
        # Compute circumcenter of the triangle on the sphere
        # The circumcenter is the normalized normal to the plane containing v1, v2, v3
        # n = (v2 - v1) x (v3 - v1)
        n = np.cross(v2 - v1, v3 - v1)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-10:
            # Degenerate triangle
            continue
            
        c = n / n_norm
        
        # Check if c is inside the spherical triangle
        # This is true if the triangle is acute-angled.
        # We can check if c is in the positive span of v1, v2, v3.
        # Solve A * coeffs = c where A = [v1, v2, v3]
        # If coeffs are all positive, c is inside.
        A = np.column_stack([v1, v2, v3])
        try:
            coeffs = np.linalg.solve(A, c)
            inside = np.all(coeffs > -1e-8)
        except np.linalg.LinAlgError:
            inside = False
            
        if inside:
            # The worst point is the circumcenter c
            # Distance from c to v1 is the radius
            dot = np.clip(np.dot(c, v1), -1.0, 1.0)
            radius = np.arccos(dot)
        else:
            # The worst point is one of the edge midpoints
            # Check midpoints of the 3 edges
            m12 = _normalize(v1 + v2)
            m23 = _normalize(v2 + v3)
            m31 = _normalize(v3 + v1)
            
            r12 = np.arccos(np.clip(np.dot(m12, v1), -1.0, 1.0))
            r23 = np.arccos(np.clip(np.dot(m23, v2), -1.0, 1.0))
            r31 = np.arccos(np.clip(np.dot(m31, v3), -1.0, 1.0))
            
            radius = max(r12, r23, r31)
            
        if radius > max_radius:
            max_radius = radius
            
    return max_radius * 180.0 / np.pi

def _local_optimize(pts, max_time=10.0):
    """
    Local optimization using Nelder-Mead on the flattened points.
    """
    start_time = time.time()
    x0 = pts.flatten()
    
    def objective(x):
        if time.time() - start_time > max_time:
            return 1e10
        p = x.reshape(N, 3)
        p = _normalize(p)
        return _covering_radius(p)
        
    # Use Nelder-Mead for non-smooth functions
    # maxiter is limited by time
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    if res.success or res.fun < objective(x0):
        best_pts = _normalize(res.x.reshape(N, 3))
        best_cr = res.fun
    else:
        best_pts = pts
        best_cr = _covering_radius(pts)
        
    return best_pts, best_cr

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Local optimization from Fibonacci
    if time.time() < deadline - 20.0:
        opt_pts, cr = _local_optimize(pts, max_time=30.0)
        if cr < best_cr:
            best_pts = opt_pts
            best_cr = cr
            
    # Try random restarts
    np.random.seed(42)
    for restart in range(10):
        if time.time() > deadline - 30.0:
            break
            
        rand_pts = np.random.randn(N, 3)
        rand_pts = _normalize(rand_pts)
        
        opt_pts, cr = _local_optimize(rand_pts, max_time=10.0)
        if cr < best_cr:
            best_pts = opt_pts
            best_cr = cr
            
    return _normalize(best_pts)
