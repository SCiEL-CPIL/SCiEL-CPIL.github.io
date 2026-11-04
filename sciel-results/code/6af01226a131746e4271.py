import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14

# Pre-compute indices for all triangles i < j < k
TRI_INDICES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area using pre-computed indices."""
    if len(points) < 3:
        return 0.0
    i, j, k = TRI_INDICES.T
    pi = points[i]
    pj = points[j]
    pk = points[k]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _convex_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _two_heptagons(r, theta):
    """
    Generate 14 points: 7 on outer unit circle, 7 on inner circle of radius r,
    with inner ring rotated by theta relative to outer ring.
    """
    n_out = 7
    n_in = 7
    angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + theta
    
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    return np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])

def _regular_polygon(n):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _local_optimize(points, time_limit=10.0):
    """Local optimization in full 2D space using Nelder-Mead."""
    start_time = time.time()
    
    def obj(x):
        pts = x.reshape(-1, 2)
        # Small penalty if points are too close to avoid numerical issues, 
        # though the score function handles it.
        return -_score(pts)
    
    x0 = points.flatten()
    best_score = _score(points)
    best_pts = points.copy()
    
    try:
        # Nelder-Mead is robust for non-smooth functions
        res = minimize(
            obj, x0, method='Nelder-Mead',
            options={
                'maxiter': 500, 
                'xatol': 1e-6, 
                'fatol': 1e-8, 
                'adaptive': True
            }
        )
        
        # Check time
        if time.time() < start_time + time_limit:
            if -res.fun > best_score:
                best_score = -res.fun
                best_pts = res.x.reshape(-1, 2).copy()
    except Exception:
        pass
    
    return best_score, best_pts

def construct():
    start_time = time.time()
    # Safety margin: 120s limit, aim for 115s
    deadline = start_time + 115.0
    
    best_score = 0.0
    best_points = None
    
    # Strategy 1: Regular 14-gon
    pts_14 = _regular_polygon(14)
    s_14 = _score(pts_14)
    if s_14 > best_score:
        best_score = s_14
        best_points = pts_14.copy()
        
    # Strategy 2: Two Heptagons (7 outer, 7 inner)
    # Coarse grid search to find a good starting point for local optimization
    # The symmetry group is D7, so theta is periodic with 2*pi/7.
    # We search r in [0, 1] and theta in [0, pi/7] (half period due to reflection symmetry)
    
    r_vals = np.linspace(0.1, 0.95, 15)
    theta_vals = np.linspace(0, np.pi / 7, 15)
    
    best_ring_score = 0.0
    best_ring_pts = None
    best_ring_r = 0.5
    best_ring_theta = 0.0
    
    for r in r_vals:
        if time.time() > deadline - 30:
            break
        for theta in theta_vals:
            pts = _two_heptagons(r, theta)
            s = _score(pts)
            if s > best_ring_score:
                best_ring_score = s
                best_ring_pts = pts.copy()
                best_ring_r = r
                best_ring_theta = theta
            if s > best_score:
                best_score = s
                best_points = pts.copy()
                
    # If we found a good ring configuration, refine it with local optimization
    if best_ring_pts is not None and time.time() < deadline - 20:
        # The grid might be too coarse. Let's do a quick 2D optimization on (r, theta) first?
        # Or just jump to full 28D Nelder-Mead from the best grid point.
        # Full 28D Nelder-Mead is expensive but can break symmetry if beneficial.
        
        s_local, pts_local = _local_optimize(best_ring_pts, time_limit=15.0)
        if s_local > best_score:
            best_score = s_local
            best_points = pts_local
            
        # Also try local optimization from the 14-gon
        if time.time() < deadline - 10:
            s_local_14, pts_local_14 = _local_optimize(pts_14, time_limit=10.0)
            if s_local_14 > best_score:
                best_score = s_local_14
                best_points = pts_local_14

    # Strategy 3: Random interior points (to explore non-symmetric solutions)
    if time.time() < deadline - 15:
        np.random.seed(42)
        # 7 on hull, 7 random inside
        hull_pts = _regular_polygon(7)
        inner_pts = np.random.uniform(-0.8, 0.8, (7, 2))
        # Ensure inner points are not too close to hull or each other
        # Simple rejection sampling
        valid_inner = []
        for _ in range(100):
            if len(valid_inner) >= 7:
                break
            p = np.random.uniform(-0.8, 0.8, 2)
            # Check distance to hull points
            dists = np.linalg.norm(hull_pts - p, axis=1)
            if np.min(dists) > 0.2:
                # Check distance to other inner points
                if valid_inner:
                    vi = np.array(valid_inner)
                    dists_v = np.linalg.norm(vi - p, axis=1)
                    if np.min(dists_v) > 0.2:
                        valid_inner.append(p)
                else:
                    valid_inner.append(p)
                    
        if len(valid_inner) == 7:
            rand_pts = np.vstack([hull_pts, np.array(valid_inner)])
            s_rand = _score(rand_pts)
            if s_rand > best_score:
                best_score = s_rand
                best_points = rand_pts.copy()
                
            # Local optimize this random start
            if time.time() < deadline - 5:
                s_rand_local, pts_rand_local = _local_optimize(rand_pts, time_limit=10.0)
                if s_rand_local > best_score:
                    best_score = s_rand_local
                    best_points = pts_rand_local

    if best_points is None:
        # Fallback
        best_points = _regular_polygon(14)
        
    return best_points
