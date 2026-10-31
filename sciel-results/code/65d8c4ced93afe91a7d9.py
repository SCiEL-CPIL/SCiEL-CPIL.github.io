import numpy as np
from scipy.spatial import ConvexHull
import time

N = 13

def compute_min_triangle_area(points):
    n = points.shape[0]
    # Generate indices for all unique triples i < j < k
    # Using np.triu_indices or manual loops?
    # For N=13, manual loops in Python are ~286 iterations.
    # Vectorized:
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
        # Check for degenerate cases
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
    # Scale to unit area
    scale = 1.0 / np.sqrt(hull_area)
    pts_scaled = points * scale
    min_tri_area = compute_min_triangle_area(pts_scaled)
    return min_tri_area

def construct():
    start_time = time.time()
    time_limit = 115.0 # 5s buffer
    
    np.random.seed(42)
    
    # Pre-compute indices for triangle areas to save time in loop
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

    # Initial points: Regular Polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    best_score = fast_score(best_points)
    
    current_points = best_points.copy()
    current_score = best_score
    
    # Simulated Annealing Parameters
    T = 0.05
    T_min = 1e-6
    cooling_rate = 0.999
    step_size = 0.05
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        # Generate neighbor
        new_points = current_points.copy()
        # Jitter a random point
        idx = np.random.randint(N)
        jitter = np.random.normal(0, step_size, 2)
        new_points[idx] += jitter
        
        new_score = fast_score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    current_points = new_points
                    current_score = new_score
            no_improve_count += 1
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Reheating strategy
        if no_improve_count > max_no_improve:
            T = 0.05
            step_size = 0.05
            no_improve_count = 0
            # Optionally restart from best or current
            # current_points = best_points.copy() # This makes it a "multistart" SA
            # For now, just reheat locally
    
    return best_points
