import numpy as np
import time
import scipy.spatial as spat
from itertools import combinations
import scipy.optimize as opt

N = 13

# Pre-compute triangle indices for N=13
_triple_idx = list(combinations(range(N), 3))
_i_arr = np.array([t[0] for t in _triple_idx])
_j_arr = np.array([t[1] for t in _triple_idx])
_k_arr = np.array([t[2] for t in _triple_idx])
_n_triples = len(_triple_idx)

def _triangle_areas(points):
    """Compute areas of all C(N,3) triangles."""
    pts = points
    v1 = pts[_j_arr] - pts[_i_arr]
    v2 = pts[_k_arr] - pts[_i_arr]
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(det)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(points) < 3:
        return 0.0
    
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0

    areas = _triangle_areas(points)
    min_area = np.min(areas)
    
    return min_area / hull_area

def _find_bottleneck_triangles(points, areas, k=5):
    """
    Find the indices of the k smallest area triangles.
    """
    if len(areas) == 0:
        return []
    # Get indices of smallest k
    idx = np.argpartition(areas, k)[:k]
    # Sort them by area for consistency
    idx = idx[np.argsort(areas[idx])]
    return list(idx)

def _bottleneck_directed_search(points, max_iter=5000, time_budget=50.0):
    """
    Directed search that tries to increase the area of the bottleneck triangles.
    For each bottleneck triangle, we try to move its vertices away from each other.
    We also check that we don't significantly decrease the area of other small triangles.
    """
    start_time = time.time()
    pts = points.copy().astype(np.float64)
    best_score = calculate_score(pts)
    best_pts = pts.copy()
    
    # Adaptive step size
    step = 0.01
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        areas = _triangle_areas(pts)
        hull_area = _hull_area(pts)
        if hull_area <= 1e-9:
            break
        
        current_score = np.min(areas) / hull_area
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
        
        # Find bottleneck triangles
        bottleneck_idx = _find_bottleneck_triangles(pts, areas, k=3)
        
        if not bottleneck_idx:
            break
            
        improved = False
        
        for tri_idx in bottleneck_idx:
            i, j, k = _triple_idx[tri_idx]
            
            # Get the triangle vertices
            pi = pts[i]
            pj = pts[j]
            pk = pts[k]
            
            # Compute centroid
            centroid = (pi + pj + pk) / 3.0
            
            # Direction to move each vertex: away from centroid
            # But we need to be careful not to break the hull or other triangles.
            # Simple strategy: move each vertex slightly away from the centroid.
            
            # Try moving one vertex at a time
            for v_idx in [i, j, k]:
                pv = pts[v_idx]
                # Vector from centroid to vertex
                dir_vec = pv - centroid
                norm = np.linalg.norm(dir_vec)
                if norm < 1e-9:
                    continue
                dir_vec = dir_vec / norm
                
                # Try a small move in that direction
                new_pts = pts.copy()
                new_pts[v_idx] = pv + step * dir_vec
                
                # Calculate new score
                new_areas = _triangle_areas(new_pts)
                new_hull_area = _hull_area(new_pts)
                if new_hull_area <= 1e-9:
                    continue
                
                new_score = np.min(new_areas) / new_hull_area
                
                if new_score > best_score * 0.99: # Allow slight decrease if it might lead to improvement later? No, strict improvement is safer.
                    # Actually, let's require improvement over current best, or at least not worse than current.
                    if new_score >= current_score:
                        # Check if the specific bottleneck triangle improved
                        new_bottleneck_area = new_areas[tri_idx]
                        old_bottleneck_area = areas[tri_idx]
                        
                        if new_bottleneck_area > old_bottleneck_area:
                            pts = new_pts
                            improved = True
                            break
            if improved:
                break
        
        if not improved:
            # If no direct improvement, try a random perturbation of a bottleneck vertex
            # Or reduce step size and try again
            step *= 0.95
            if step < 1e-5:
                # Try random restarts from current best
                # Or just break
                break
        
        # Occasionally try a larger jump
        if it % 100 == 0:
            step = 0.05 * (0.5 + 0.5 * np.random.rand())
            
    return best_pts, best_score

def _local_polish(points, max_iter=2000, time_budget=10.0):
    """
    Fine-tune the points using a local search that minimizes the variance of small triangle areas
    or simply maximizes the minimum area.
    """
    start_time = time.time()
    pts = points.copy().astype(np.float64)
    best_score = calculate_score(pts)
    best_pts = pts.copy()
    
    step = 0.005
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
            
        areas = _triangle_areas(pts)
        hull_area = _hull_area(pts)
        if hull_area <= 1e-9:
            break
            
        current_score = np.min(areas) / hull_area
        
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()
            
        # Find the smallest triangle
        min_idx = np.argmin(areas)
        i, j, k = _triple_idx[min_idx]
        
        # Try to increase the area of this triangle by moving vertices
        # Gradient of area wrt pi: 0.5 * (pj - pk) rotated 90 deg?
        # Area = 0.5 * |det(pj-pi, pk-pi)|
        # Let's just try small random moves on the 3 vertices
        improved = False
        for _ in range(10):
            trial = pts.copy()
            # Move one of the vertices randomly
            v_idx = np.random.choice([i, j, k])
            trial[v_idx] += step * np.random.randn(2)
            
            trial_areas = _triangle_areas(trial)
            trial_hull = _hull_area(trial)
            if trial_hull <= 1e-9:
                continue
            trial_score = np.min(trial_areas) / trial_hull
            
            if trial_score > current_score:
                pts = trial
                improved = True
                break
        
        if not improved:
            step *= 0.9
            if step < 1e-6:
                break
                
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = -1.0
    best_points = None
    
    # Strategy:
    # 1. Generate diverse initial configurations (regular polygons, rings, random)
    # 2. For each, run bottleneck-directed search
    # 3. Polish the best result
    
    # Initial guesses
    initial_configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    initial_configs.append(np.column_stack((np.cos(angles), np.sin(angles))))
    
    # 2. Ring configurations
    # 1 center, 12 outer
    center = np.array([[0, 0]])
    angles12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    outer12 = np.column_stack((np.cos(angles12), np.sin(angles12)))
    initial_configs.append(np.vstack([center, outer12]))
    
    # 3. 2 inner, 11 outer
    angles2 = np.linspace(0, 2*np.pi, 2, endpoint=False)
    inner2 = 0.5 * np.column_stack((np.cos(angles2), np.sin(angles2)))
    angles11 = np.linspace(0, 2*np.pi, 11, endpoint=False)
    outer11 = np.column_stack((np.cos(angles11), np.sin(angles11)))
    initial_configs.append(np.vstack([inner2, outer11]))
    
    # 4. 3 inner, 10 outer
    angles3 = np.linspace(0, 2*np.pi, 3, endpoint=False)
    inner3 = 0.6 * np.column_stack((np.cos(angles3), np.sin(angles3)))
    angles10 = np.linspace(0, 2*np.pi, 10, endpoint=False)
    outer10 = np.column_stack((np.cos(angles10), np.sin(angles10)))
    initial_configs.append(np.vstack([inner3, outer10]))
    
    # 5. 4 inner, 9 outer
    angles4 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    inner4 = 0.7 * np.column_stack((np.cos(angles4), np.sin(angles4)))
    angles9 = np.linspace(0, 2*np.pi, 9, endpoint=False)
    outer9 = np.column_stack((np.cos(angles9), np.sin(angles9)))
    initial_configs.append(np.vstack([inner4, outer9]))
    
    # 6. Random points in unit disk
    for _ in range(5):
        r = np.sqrt(np.random.rand(N))
        theta = np.random.rand(N) * 2 * np.pi
        random_pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        initial_configs.append(random_pts)
        
    # Evaluate and search for each initial config
    for i, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 20.0:
            break
            
        # Time budget for this config
        time_budget = min(15.0, deadline - time.time() - 20.0)
        if time_budget < 2.0:
            break
            
        try:
            # Step 1: Bottleneck directed search
            pts, score = _bottleneck_directed_search(init_pts, time_budget=time_budget * 0.8)
            
            # Step 2: Local polish
            if time.time() < deadline - 10.0:
                polished_pts, polished_score = _local_polish(pts, time_budget=min(5.0, deadline - time.time() - 10.0))
                if polished_score > score:
                    pts, score = polished_pts, polished_score
                    
            if score > best_score:
                best_score = score
                best_points = pts
        except Exception as e:
            pass
    
    # If we didn't find anything good, fallback to regular 13-gon
    if best_score < 0.1 or best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = calculate_score(best_points)
        
    # Final global polish on the best points found
    if best_points is not None and time.time() < deadline - 5.0:
        final_polished, final_score = _local_polish(best_points, time_budget=min(10.0, deadline - time.time() - 5.0))
        if final_score > best_score:
            best_score = final_score
            best_points = final_polished
            
    if best_points is None:
        return np.random.rand(N, 2)
        
    return best_points
