import numpy as np
import time
from scipy.spatial import ConvexHull

N = 14

def _get_min_triple_and_score(points):
    """
    Returns (min_area, hull_area, min_triple_indices)
    """
    n = points.shape[0]
    min_area = np.inf
    min_triple = (0, 1, 2)
    
    # Calculate all triangle areas
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = abs(cross) * 0.5
                if area < min_area:
                    min_area = area
                    min_triple = (i, j, k)
    
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0, 0.0, min_triple
        
    if hull_area < 1e-9:
        return 0.0, 0.0, min_triple
        
    return min_area, hull_area, min_triple

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack((x, y))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initial guess: Regular 14-gon
    best_points = _regular_polygon(N)
    min_area, hull_area, _ = _get_min_triple_and_score(best_points)
    best_score = min_area / hull_area if hull_area > 0 else 0.0
    
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # Identify the critical triple
        _, _, critical_triple = _get_min_triple_and_score(best_points)
        
        # Generate candidates
        candidates = []
        
        # 1. Perturb all points slightly
        noise = np.random.normal(0, step_size, best_points.shape)
        candidates.append(best_points + noise)
        
        # 2. Perturb only the critical triple points
        for pt_idx in critical_triple:
            for _ in range(5): # Try a few random moves for this point
                cand = best_points.copy()
                noise = np.random.normal(0, step_size * 2, 2)
                cand[pt_idx] += noise
                candidates.append(cand)
                
        # 3. Try to move points away from the center if they are too close?
        # Or just rely on the above.
        
        # Evaluate candidates
        best_candidate = best_points
        best_candidate_score = best_score
        
        for cand in candidates:
            # Basic validity check: points should be distinct
            # and hull should be valid
            min_a, hull_a, _ = _get_min_triple_and_score(cand)
            if hull_a > 1e-9:
                score = min_a / hull_a
                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate = cand
        
        if best_candidate_score > best_score:
            best_points = best_candidate
            best_score = best_candidate_score
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            # Restart or increase step
            step_size *= 1.5
            if step_size > 0.5:
                # Random restart from regular polygon with perturbation
                angle_offset = np.random.uniform(0, 2 * np.pi)
                best_points = _regular_polygon(N, radius=1.0)
                theta = angle_offset
                c, s = np.cos(theta), np.sin(theta)
                rot = np.array([[c, -s], [s, c]])
                best_points = best_points @ rot.T
                noise = np.random.normal(0, 0.1, best_points.shape)
                best_points = best_points + noise
                min_a, hull_a, _ = _get_min_triple_and_score(best_points)
                best_score = min_a / hull_a if hull_a > 0 else 0.0
                step_size = 0.05
                no_improve_count = 0

    return best_points
