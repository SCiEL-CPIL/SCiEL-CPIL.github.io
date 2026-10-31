import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 13

def _compute_min_triangle_area(points, i_idx, j_idx, k_idx):
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _compute_hull_area(points):
    try:
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    np.random.seed(42)
    
    # Pre-compute indices for triangle areas
    n = N
    i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    mask = (i < j) & (j < k)
    i_idx = i[mask]
    j_idx = j[mask]
    k_idx = k[mask]
    
    def fast_score(points):
        hull_area = _compute_hull_area(points)
        if hull_area < 1e-10:
            return 0.0
        scale = 1.0 / np.sqrt(hull_area)
        pts_scaled = points * scale
        return _compute_min_triangle_area(pts_scaled, i_idx, j_idx, k_idx)
    
    def objective(x):
        # x is a flat array of shape (2*N,)
        points = x.reshape(N, 2)
        s = fast_score(points)
        # We want to maximize s, so minimize -s
        # Add a small penalty if points are too close to avoid numerical issues in hull
        return -s

    # Multi-Start Nelder-Mead
    # Generate diverse initial points
    initial_points_list = []
    
    # 1. Regular Polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_points_list.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Regular Polygon with slight rotation
    angles_rot = angles + np.pi / N
    initial_points_list.append(np.column_stack([np.cos(angles_rot), np.sin(angles_rot)]))
    
    # 3. Random points in unit circle
    for _ in range(5):
        r = np.sqrt(np.random.rand(N))
        theta = np.random.rand(N) * 2 * np.pi
        initial_points_list.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
        
    # 4. Points on a grid (subset)
    grid = np.linspace(-1, 1, 4)
    grid_points = []
    for gx in grid:
        for gy in grid:
            grid_points.append([gx, gy])
    # Shuffle and take 13
    np.random.shuffle(grid_points)
    initial_points_list.append(np.array(grid_points[:N]))
    
    best_points = initial_points_list[0]
    best_score = fast_score(best_points)
    
    # Optimize each initial point
    for init_pts in initial_points_list:
        if time.time() - start_time > time_limit - 5:
            break
            
        x0 = init_pts.flatten()
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.success or res.fun < objective(x0):
                res_pts = res.x.reshape(N, 2)
                res_score = fast_score(res_pts)
                
                if res_score > best_score:
                    best_score = res_score
                    best_points = res_pts
        except Exception:
            continue
            
    return best_points
