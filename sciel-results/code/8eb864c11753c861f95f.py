import numpy as np
import time
import scipy.spatial as spat
import itertools

N = 13

def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(points) < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = spat.ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    pts = points
    n = len(pts)
    
    # Efficient triangle area calculation
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    v1 = pts[j_flat] - pts[i_flat] 
    v2 = pts[k_flat] - pts[i_flat]
    
    # Determinant in 2D
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(det)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def find_bottleneck(points):
    """
    Returns the indices (i, j, k) of the triangle with the minimum area,
    and the area itself.
    """
    pts = points
    n = len(pts)
    
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    v1 = pts[j_flat] - pts[i_flat] 
    v2 = pts[k_flat] - pts[i_flat]
    
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(det)
    
    min_idx = np.argmin(areas)
    return i_flat[min_idx], j_flat[min_idx], k_flat[min_idx], areas[min_idx]

def _local_polish(points, max_iter=200, deadline=None):
    """
    Local refinement using a simple hill-climbing step on the bottleneck triangle.
    """
    current_pts = points.copy()
    current_score = calculate_score(current_pts)
    step_size = 0.05
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        i, j, k, _ = find_bottleneck(current_pts)
        
        p_i = current_pts[i]
        p_j = current_pts[j]
        p_k = current_pts[k]
        
        centroid_jk = (p_j + p_k) / 2.0
        centroid_ik = (p_i + p_k) / 2.0
        centroid_ij = (p_i + p_j) / 2.0
        
        dir_i = p_i - centroid_jk
        dir_j = p_j - centroid_ik
        dir_k = p_k - centroid_ij
        
        norm_i = np.linalg.norm(dir_i)
        norm_j = np.linalg.norm(dir_j)
        norm_k = np.linalg.norm(dir_k)
        
        if norm_i > 1e-6: dir_i /= norm_i
        if norm_j > 1e-6: dir_j /= norm_j
        if norm_k > 1e-6: dir_k /= norm_k
        
        trial_pts = current_pts.copy()
        trial_pts[i] += step_size * dir_i
        trial_pts[j] += step_size * dir_j
        trial_pts[k] += step_size * dir_k
        
        trial_score = calculate_score(trial_pts)
        
        if trial_score > current_score:
            current_pts = trial_pts
            current_score = trial_score
        else:
            step_size *= 0.9
            if step_size < 1e-4:
                break
                
    return current_pts, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_score = -1.0
    best_points = None
    
    # 1. Deterministic Geometric Construction
    # This replaces random initialization with a structure known to maximize
    # minimum distances/areas for small N.
    
    # Hexagon (6 points)
    hex_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    hex_pts = np.column_stack((np.cos(hex_angles), np.sin(hex_angles)))
    
    # Inner circle (5 points)
    inner_angles = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5 # Offset
    inner_pts = 0.5 * np.column_stack((np.cos(inner_angles), np.sin(inner_angles)))
    
    # Center (1 point)
    center = np.array([[0.0, 0.0]])
    
    # Initial configuration
    init_pts = np.vstack([hex_pts, inner_pts, center])
    
    # 2. Structured Local Search
    # Instead of generic DE, we perform a focused search on this good starting point.
    
    current_pts, current_score = _local_polish(init_pts, max_iter=50, deadline=deadline)
    best_score = current_score
    best_points = current_pts.copy()
    
    # 3. Multi-start with geometric variations to escape local optima
    # We vary the radius of the inner circle and the rotation.
    
    radii = [0.4, 0.5, 0.6, 0.7]
    rotations = [0, np.pi/5, np.pi/10]
    
    for r in radii:
        for rot in rotations:
            if time.time() > deadline - 10.0:
                break
            
            # Reconstruct with new parameters
            new_hex = hex_pts # Keep hexagon fixed for now, or rotate slightly
            new_inner = r * np.column_stack((np.cos(inner_angles + rot), np.sin(inner_angles + rot)))
            new_center = center
            
            trial_init = np.vstack([new_hex, new_inner, new_center])
            
            # Polish
            polished_pts, polished_score = _local_polish(trial_init, max_iter=50, deadline=deadline)
            
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_pts.copy()

    # 4. Final Polish
    if best_points is not None and time.time() < deadline - 5.0:
        polished_pts, polished_score = _local_polish(best_points, max_iter=100, deadline=deadline - 5.0)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts
            
    if best_points is None:
        return np.random.rand(N, 2)
        
    return best_points
