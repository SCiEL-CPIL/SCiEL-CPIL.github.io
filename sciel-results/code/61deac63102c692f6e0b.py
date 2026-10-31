import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 13

def compute_min_triangle_area(points):
    n = points.shape[0]
    i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    mask = (i < j) & (j < k)
    i_idx = i[mask]
    j_idx = j[mask]
    k_idx = k[mask]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    if len(areas) == 0:
        return 0.0
    return np.min(areas)

def compute_hull_area(points):
    try:
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def score(points):
    hull_area = compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    scale = 1.0 / np.sqrt(hull_area)
    pts_scaled = points * scale
    min_tri_area = compute_min_triangle_area(pts_scaled)
    return min_tri_area

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

    def fast_min_tri_area(points):
        p_i = points[i_idx]
        p_j = points[j_idx]
        p_k = points[k_idx]
        v1 = p_j - p_i
        v2 = p_k - p_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        return np.min(areas)

    def fast_score(points):
        hull_area = compute_hull_area(points)
        if hull_area < 1e-10:
            return 0.0
        scale = 1.0 / np.sqrt(hull_area)
        pts_scaled = points * scale
        return fast_min_tri_area(pts_scaled)

    def objective(x):
        # x is flat array of length 2*N
        pts = x.reshape(N, 2)
        s = fast_score(pts)
        return -s # We want to maximize, so minimize negative

    best_points = None
    best_score = -np.inf

    # Generate diverse initial candidates
    initial_candidates = []
    
    # 1. Regular Polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_poly = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_candidates.append(reg_poly)
    
    # 2. Random Points in Unit Circle
    for _ in range(5):
        r = np.sqrt(np.random.rand(N))
        theta = 2 * np.pi * np.random.rand(N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        initial_candidates.append(pts)
        
    # 3. Regular Polygon with Noise
    for noise_level in [0.01, 0.05, 0.1]:
        noisy = reg_poly + np.random.normal(0, noise_level, (N, 2))
        initial_candidates.append(noisy)

    # 4. "Clustered" initial guess: Points on two concentric circles
    # This helps break symmetry of the regular polygon
    n1 = N // 2
    n2 = N - n1
    r1, r2 = 0.5, 1.0
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(np.pi/n2, 2 * np.pi, n2, endpoint=False) # offset
    p1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
    p2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
    clustered = np.vstack([p1, p2])
    initial_candidates.append(clustered)
    
    # Shuffle candidates
    np.random.shuffle(initial_candidates)

    # Run Nelder-Mead on each candidate until time limit
    for init_pts in initial_candidates:
        if time.time() - start_time > time_limit:
            break
            
        x0 = init_pts.flatten()
        
        # Use Nelder-Mead with adaptive settings
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            
            current_pts = res.x.reshape(N, 2)
            current_score = fast_score(current_pts)
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_pts.copy()
                
        except Exception:
            continue

    # If no valid points found (should not happen), fallback to regular polygon
    if best_points is None:
        best_points = initial_candidates[0]

    return best_points
