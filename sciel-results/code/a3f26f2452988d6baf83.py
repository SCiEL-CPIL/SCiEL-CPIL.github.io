import numpy as np
from scipy.spatial import ConvexHull
import time

N = 13

def _score_points(pts):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    n = len(pts)
    if n < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 0:
            return 0.0
    except Exception:
        return 0.0
        
    # Compute all triangle areas
    min_area = float('inf')
    # Vectorized calculation for speed
    # We can't easily vectorize the triple loop without huge memory usage for N=13?
    # N=13, 286 triangles. Loop is fine.
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            pj = pts[j]
            v1 = pj - pi
            for k in range(j + 1, n):
                pk = pts[k]
                v2 = pk - pi
                # Cross product magnitude in 2D
                cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
                    
    if min_area == float('inf'):
        return 0.0
        
    return min_area / hull_area

def _generate_candidate(config_type):
    """
    Generate a specific symmetric configuration.
    """
    pts = []
    if config_type == 'reg13':
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
    elif config_type == 'reg12_center':
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        ring1 = np.column_stack([np.cos(angles), np.sin(angles)])
        center = np.array([[0.0, 0.0]])
        pts = np.vstack([ring1, center])
    elif config_type == 'reg11_center_1':
        # 11 on unit circle, 1 at center, 1 at radius 0.5
        angles11 = np.linspace(0, 2 * np.pi, 11, endpoint=False)
        ring1 = np.column_stack([np.cos(angles11), np.sin(angles11)])
        center = np.array([[0.0, 0.0]])
        p_extra = np.array([[0.5, 0.0]])
        pts = np.vstack([ring1, center, p_extra])
    elif config_type == 'reg10_center_2':
        # 10 on unit circle, 1 at center, 2 at radius 0.5 (opposite)
        angles10 = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        ring1 = np.column_stack([np.cos(angles10), np.sin(angles10)])
        center = np.array([[0.0, 0.0]])
        p1 = np.array([[0.5, 0.0]])
        p2 = np.array([[-0.5, 0.0]])
        pts = np.vstack([ring1, center, p1, p2])
    else:
        return None
        
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # 100s limit
    
    candidates = ['reg13', 'reg12_center', 'reg11_center_1', 'reg10_center_2']
    best_pts = None
    best_score = -1.0
    
    for c in candidates:
        if time.time() > deadline:
            break
        pts = _generate_candidate(c)
        if pts is None:
            continue
        score = _score_points(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
            
    # If we have time, try to optimize the best candidate slightly?
    # For now, return the best static candidate.
    # The random seed score is very low, so any of these will be a massive improvement.
    
    if best_pts is None:
        # Fallback to random if something went wrong
        return np.random.rand(N, 2)
        
    return best_pts
