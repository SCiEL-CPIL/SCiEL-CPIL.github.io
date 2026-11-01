import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares

N = 130
_EPS = 1e-12
_DEADLINE_MARGIN = 5.0  # Safety margin for the 150s limit

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(rng):
    """Generate random initial points."""
    pts = rng.standard_normal((N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _compute_hull_data(pts):
    """
    Compute convex hull and return data needed for radius calculation and optimization.
    Returns: (hull, faces, A, B, C, normals, cos_thetas)
    If hull computation fails, returns None.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return None
            
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        return (hull, faces[valid], A[valid], B[valid], C[valid], normals, cos_theta)
    except Exception:
        return None

def _compute_hull_radius(pts):
    """Compute the exact covering radius in radians."""
    data = _compute_hull_data(pts)
    if data is None:
        return np.pi
    _, _, _, _, _, _, cos_theta = data
    ang_radii = np.arccos(cos_theta)
    return np.max(ang_radii)

def _lloyd_relax(pts, max_iters=50, tol=1e-6):
    """
    Perform Lloyd relaxation: move each point to the centroid of its Voronoi cell.
    For points on the sphere, the Voronoi cell is approximated by the convex hull facets.
    """
    current_pts = pts.copy()
    
    for _ in range(max_iters):
        data = _compute_hull_data(current_pts)
        if data is None:
            break
            
        hull, faces, A, B, C, normals, cos_theta = data
        
        # For each point, find the average normal of the facets it belongs to
        # This normal points outward. The "centroid" of the Voronoi cell on the sphere
        # is in the direction of the sum of the outward normals of the adjacent facets,
        # weighted by the facet area.
        
        # Map point index to list of facet indices
        # faces shape is (F, 3), each row is [i, j, k]
        
        new_pts = np.zeros_like(current_pts)
        weights = np.zeros(N)
        
        # Calculate area weights
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = np.linalg.norm(cross, axis=1)
        
        # Accumulate normals for each point
        for f_idx, face in enumerate(faces):
            n = normals[f_idx]
            w = area[f_idx]
            
            for p_idx in face:
                new_pts[p_idx] += w * n
                weights[p_idx] += w
        
        # Normalize new points to unit sphere
        # If a point has no facets (should not happen for convex hull), keep original
        valid_w = weights > _EPS
        new_pts[valid_w] /= np.linalg.norm(new_pts[valid_w], axis=1, keepdims=True)
        new_pts[~valid_w] = current_pts[~valid_w]
        
        # Check convergence
        diff = np.linalg.norm(new_pts - current_pts, axis=1)
        current_pts = new_pts
        if np.max(diff) < tol:
            break
            
    return current_pts

def _residuals(x, pts_shape):
    """
    Residuals for least_squares.
    We want to minimize the maximum circumradius.
    A common trick is to minimize the sum of squares of (radius - target) for radii > target,
    or simply minimize the sum of squares of all radii, which tends to equalize them.
    Here, we minimize the sum of squares of the angular radii of the hull facets.
    This drives the configuration towards equalizing the circumradii.
    """
    pts = x.reshape(pts_shape)
    data = _compute_hull_data(pts)
    if data is None:
        return np.ones(100) * 100 # Large penalty
    
    _, _, _, _, _, _, cos_theta = data
    ang_radii = np.arccos(cos_theta)
    
    # To focus on the maximum, we can use a soft-max or just sum of squares.
    # Sum of squares of radii is a good proxy for minimizing the max radius in symmetric problems.
    return ang_radii

def _least_squares_polish(pts, max_nfev=1000):
    """
    Polish the configuration using least_squares.
    """
    x0 = pts.flatten()
    
    try:
        res = least_squares(
            _residuals,
            x0,
            args=(pts.shape,),
            method='trf', # Trust Region Reflective, good for bounds
            bounds=([-1.0]*N*3, [1.0]*N*3), # Points must be on unit sphere, so coords in [-1, 1]
            max_nfev=max_nfev,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10
        )
        
        # Reshape and normalize
        new_pts = res.x.reshape(pts.shape)
        new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
        
        # Check if it's valid (finite, etc.)
        if not np.all(np.isfinite(new_pts)):
            return pts
            
        return new_pts
    except Exception:
        return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_pts = _initial_fib()
    best_r = _compute_hull_radius(best_pts)
    
    # Generate multiple initial configurations
    inits = [best_pts]
    for _ in range(10):
        inits.append(_random_init(rng))
    
    for init_pts in inits:
        if time.time() > deadline:
            break
            
        # Phase 1: Lloyd Relaxation
        relaxed_pts = _lloyd_relax(init_pts, max_iters=100)
        r_relaxed = _compute_hull_radius(relaxed_pts)
        
        if r_relaxed < best_r:
            best_r = r_relaxed
            best_pts = relaxed_pts.copy()
            
        # Phase 2: Least Squares Polish
        # Only polish if we have time
        if time.time() < deadline - 10.0:
            polished_pts = _least_squares_polish(relaxed_pts, max_nfev=500)
            r_polished = _compute_hull_radius(polished_pts)
            
            if r_polished < best_r:
                best_r = r_polished
                best_pts = polished_pts.copy()
                
        # If we improved significantly, we might want to try relaxing again from the polished point?
        # Maybe just one pass is enough for the time budget.
        
    # Final normalization
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    # Validity check
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Covering radius: {np.degrees(_compute_hull_radius(pts)):.6f} degrees")
