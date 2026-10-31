import numpy as np
import time
import scipy.spatial as spat
from scipy.optimize import minimize
from itertools import combinations

N = 13
# Pre-compute triangle indices
_triple_idx = list(combinations(range(N), 3))
_i_arr = np.array([t[0] for t in _triple_idx])
_j_arr = np.array([t[1] for t in _triple_idx])
_k_arr = np.array([t[2] for t in _triple_idx])

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
        return 1e-9

def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    """
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area

def _barycentric_coords(p, triangle):
    """
    Compute barycentric coordinates of point p with respect to triangle vertices.
    Returns (u, v, w) such that p = u*t0 + v*t1 + w*t2, u+v+w=1.
    """
    t0, t1, t2 = triangle
    v0 = t1 - t0
    v1 = t2 - t0
    v2 = p - t0
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d02 = np.dot(v0, v2)
    d11 = np.dot(v1, v1)
    d12 = np.dot(v1, v2)
    
    denom = d00 * d11 - d01 * d01
    if np.abs(denom) < 1e-12:
        return None
        
    v = (d11 * d02 - d01 * d12) / denom
    w = (d00 * d12 - d01 * d02) / denom
    u = 1.0 - v - w
    
    return u, v, w

def _move_point_towards_centroid(points, i, step_size):
    """
    Move point i towards the centroid of the other N-1 points.
    This helps keep points in a balanced configuration.
    """
    others = np.delete(points, i, axis=0)
    centroid = np.mean(others, axis=0)
    new_point = points[i] + step_size * (centroid - points[i])
    new_points = points.copy()
    new_points[i] = new_point
    return new_points

def _local_search(points, time_budget):
    """
    Perform local search by iteratively moving points to improve the minimum triangle area.
    Uses a greedy approach: for each point, try small perturbations and keep the best.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    current_points = points.copy()
    current_score = calculate_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    # Pre-compute all triangles for efficiency
    # We'll use a simple hill-climbing strategy with adaptive step size
    
    step_size = 0.05
    no_improve_count = 0
    
    while time.time() < deadline - 1.0:
        improved = False
        
        for i in range(N):
            if time.time() > deadline - 1.0:
                break
            
            # Try moving point i in multiple directions
            # Directions: towards centroid, random directions, towards other points
            directions = []
            
            # Direction towards centroid of others
            others = np.delete(current_points, i, axis=0)
            centroid = np.mean(others, axis=0)
            dir_centroid = centroid - current_points[i]
            norm = np.linalg.norm(dir_centroid)
            if norm > 1e-6:
                directions.append(dir_centroid / norm)
            
            # Random directions
            for _ in range(4):
                rand_dir = np.random.randn(2)
                rand_dir /= np.linalg.norm(rand_dir)
                directions.append(rand_dir)
            
            # Directions towards each other point
            for j in range(N):
                if j != i:
                    dir_j = current_points[j] - current_points[i]
                    norm_j = np.linalg.norm(dir_j)
                    if norm_j > 1e-6:
                        directions.append(dir_j / norm_j)
            
            # Try each direction with different step sizes
            best_local_score = current_score
            best_local_point = current_points[i].copy()
            
            for dir in directions:
                for step_mult in [0.5, 1.0, 2.0]:
                    step = step_size * step_mult
                    new_point = current_points[i] + step * dir
                    new_points = current_points.copy()
                    new_points[i] = new_point
                    
                    new_score = calculate_score(new_points)
                    
                    if new_score > best_local_score:
                        best_local_score = new_score
                        best_local_point = new_point
            
            # Apply the best improvement found for point i
            if best_local_score > current_score + 1e-10:
                current_points[i] = best_local_point
                current_score = best_local_score
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
        
        if not improved:
            # Reduce step size
            step_size *= 0.7
            no_improve_count += 1
            
            if step_size < 1e-4 or no_improve_count > 20:
                break
        else:
            no_improve_count = 0
            # Slightly increase step size if improving
            step_size = min(step_size * 1.1, 0.1)
    
    return best_points, best_score

def _generate_initial_configs():
    """
    Generate diverse initial configurations.
    """
    configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    configs.append(np.column_stack((np.cos(angles), np.sin(angles))))
    
    # 2. Two-ring: 1 center, 12 outer
    center = np.array([[0, 0]])
    angles12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    outer12 = np.column_stack((np.cos(angles12), np.sin(angles12)))
    configs.append(np.vstack([center, outer12]))
    
    # 3. Two-ring: 2 inner, 11 outer
    angles2 = np.linspace(0, 2*np.pi, 2, endpoint=False)
    inner2 = 0.5 * np.column_stack((np.cos(angles2), np.sin(angles2)))
    angles11 = np.linspace(0, 2*np.pi, 11, endpoint=False)
    outer11 = np.column_stack((np.cos(angles11), np.sin(angles11)))
    configs.append(np.vstack([inner2, outer11]))
    
    # 4. Two-ring: 3 inner, 10 outer
    angles3 = np.linspace(0, 2*np.pi, 3, endpoint=False)
    inner3 = 0.6 * np.column_stack((np.cos(angles3), np.sin(angles3)))
    angles10 = np.linspace(0, 2*np.pi, 10, endpoint=False)
    outer10 = np.column_stack((np.cos(angles10), np.sin(angles10)))
    configs.append(np.vstack([inner3, outer10]))
    
    # 5. Random perturbations of regular polygon
    for _ in range(3):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = 1.0 + 0.1 * np.random.randn(N)
        angles += 0.05 * np.random.randn(N)
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        configs.append(pts)
        
    # 6. Random points in disk
    for _ in range(3):
        r = np.sqrt(np.random.rand(N))
        theta = np.random.rand(N) * 2 * np.pi
        pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        configs.append(pts)
        
    # 7. Three-ring configuration
    angles_inner = np.linspace(0, 2*np.pi, 3, endpoint=False)
    inner_ring = 0.3 * np.column_stack((np.cos(angles_inner), np.sin(angles_inner)))
    angles_mid = np.linspace(0, 2*np.pi, 5, endpoint=False) + 0.2
    mid_ring = 0.65 * np.column_stack((np.cos(angles_mid), np.sin(angles_mid)))
    angles_outer = np.linspace(0, 2*np.pi, 5, endpoint=False) + 0.4
    outer_ring = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
    configs.append(np.vstack([inner_ring, mid_ring, outer_ring]))
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = -1.0
    best_points = None
    
    configs = _generate_initial_configs()
    
    n_configs = len(configs)
    
    for i, init_pts in enumerate(configs):
        # Check time
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
            
        # Allocate time for this config
        time_for_config = min(15.0, remaining - 5.0)
        
        try:
            # First, do a quick local search
            pts, score = _local_search(init_pts, time_for_config)
            
            if score > best_score:
                best_score = score
                best_points = pts
        except Exception:
            continue
            
    # If we have time left, do a final polish on the best point found
    if best_points is not None:
        remaining = deadline - time.time()
        if remaining > 10.0:
            try:
                # Use a very fine local search for final polish
                pts_final, score_final = _local_search(best_points, remaining - 5.0)
                
                if score_final > best_score:
                    best_score = score_final
                    best_points = pts_final
            except Exception:
                pass

    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(calculate_score(pts))
