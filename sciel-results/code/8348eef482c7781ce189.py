import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares

N = 130
_EPS = 1e-12
_DEADLINE = 145.0

def _initial_fib():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_hull_data(pts):
    """
    Compute per-facet covering radius and outward pole direction.
    Returns (radii, normals, faces, valid_mask).
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        normals = cross[valid].copy()
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        return radii, normals, faces[valid], valid
    except Exception:
        return np.array([]), np.array([]), np.array([]), np.array([])

def _hull_radius(pts):
    """
    Compute the exact covering radius using the convex hull.
    Returns the maximum angular distance from the sphere to the nearest point.
    """
    radii, _, _, _ = _compute_hull_data(pts)
    if len(radii) == 0:
        return np.pi
    return radii.max()

def _lloyd_relax(pts, n_iter=50, deadline=None):
    """
    Lloyd relaxation: each point moves to the centroid of its Voronoi cell (approximated by hull facets).
    """
    pts = pts.astype(float).copy()
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        radii, normals, faces, valid = _compute_hull_data(pts)
        if len(radii) == 0:
            break
        
        M = len(faces)
        new_pts = pts.copy()
        
        cell_sum = np.zeros((N, 3))
        cell_count = np.zeros(N)
        
        for f in range(M):
            centroid = (pts[faces[f, 0]] + pts[faces[f, 1]] + pts[faces[f, 2]]) / 3.0
            centroid = centroid + normals[f]
            centroid = centroid / (np.linalg.norm(centroid) + _EPS)
            
            for v in faces[f]:
                cell_sum[v] += centroid
                cell_count[v] += 1
        
        valid_pts = cell_count > 0
        new_pts[valid_pts] = cell_sum[valid_pts] / cell_count[valid_pts, np.newaxis]
        new_pts[valid_pts] = new_pts[valid_pts] / (np.linalg.norm(new_pts[valid_pts], axis=1, keepdims=True) + _EPS)
        
        diff = np.linalg.norm(new_pts - pts, axis=1)
        pts = new_pts
        if np.max(diff) < 1e-10:
            break
    
    return pts

def _least_squares_polish(pts, deadline, max_iter_outer=50):
    """
    Iteratively tighten the target radius and use least_squares to minimize the excess circumradius of facets.
    """
    pts = pts.astype(float).copy()
    best_r = _hull_radius(pts)
    best_pts = pts.copy()
    
    # Start with a target slightly below current best to force improvement
    current_target = best_r * 0.99
    
    for outer_iter in range(max_iter_outer):
        if time.time() > deadline - 5.0:
            break
        
        # Define residual function for least_squares
        def residuals(x):
            # Reshape x back to points
            p = x.reshape(N, 3)
            # Normalize
            p = p / (np.linalg.norm(p, axis=1, keepdims=True) + _EPS)
            
            radii, _, _, _ = _compute_hull_data(p)
            if len(radii) == 0:
                return np.ones(1) * 100.0
            
            # Residuals: how much each facet exceeds the target
            # We only care about facets that are too large
            excess = np.maximum(radii - current_target, 0.0)
            
            # To make the problem well-posed for least_squares, we might need to penalize
            # all facets slightly, or just the worst ones. 
            # Let's penalize all facets that are above a lower bound to maintain structure,
            # but primarily drive down the max.
            # A common trick: minimize sum of (r_i - target)^2 for r_i > target, 
            # but also add a small term for all to prevent collapse.
            
            # Let's try: residual = max(0, r_i - target) + 0.01 * r_i
            # This encourages reducing all radii while heavily penalizing those above target.
            
            return excess + 0.01 * radii
        
        x0 = pts.flatten()
        
        try:
            res = least_squares(
                residuals, 
                x0, 
                method='trf', 
                max_nfev=1000,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8
            )
        except Exception:
            break
        
        new_pts = res.x.reshape(N, 3)
        new_pts = new_pts / (np.linalg.norm(new_pts, axis=1, keepdims=True) + _EPS)
        
        new_r = _hull_radius(new_pts)
        
        if new_r < best_r:
            best_r = new_r
            best_pts = new_pts.copy()
            # Tighten target if we improved
            current_target = new_r * 0.995
        else:
            # If no improvement, relax target slightly and try again or break
            # If we can't get below current target, the target might be too aggressive
            # or we are in a local minimum.
            # Let's try a slightly higher target to see if we can move at all
            current_target = best_r * 0.999
            if outer_iter > 10:
                break
    
    return best_pts, best_r

def _construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # 1. Initial point: Fibonacci spiral
    pts_init = _initial_fib()
    
    # 2. Lloyd relaxation to get a good starting structure
    pts_lloyd = _lloyd_relax(pts_init, n_iter=50, deadline=deadline - 60.0)
    r_lloyd = _hull_radius(pts_lloyd)
    
    best_pts = pts_lloyd.copy()
    best_r = r_lloyd
    
    # 3. Least squares polish
    if time.time() < deadline - 30.0:
        pts_ls, r_ls = _least_squares_polish(pts_lloyd, deadline - 10.0, max_iter_outer=30)
        if r_ls < best_r:
            best_r = r_ls
            best_pts = pts_ls.copy()
    
    # 4. One more Lloyd pass from the LS result to fix any structural issues
    if time.time() < deadline - 10.0:
        pts_final = _lloyd_relax(best_pts, n_iter=20, deadline=deadline - 5.0)
        r_final = _hull_radius(pts_final)
        if r_final < best_r:
            best_r = r_final
            best_pts = pts_final.copy()
    
    # 5. Final normalization
    best_pts = np.array(best_pts, dtype=float)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / (norms + _EPS)
    
    # Validity check
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
    
    return best_pts

def construct():
    return _construct()
