import numpy as np
import time

N = 11

def is_inside_triangle(p):
    x, y = p
    if x < -1e-9 or y < -1e-9:
        return False
    if y > np.sqrt(3) * (1 - x) + 1e-9:
        return False
    return True

def init_areas(points):
    areas = np.zeros((N, N, N))
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1, p2, p3 = points[i], points[j], points[k]
                v1 = p2 - p1
                v2 = p3 - p1
                det = v1[0]*v2[1] - v1[1]*v2[0]
                area = 0.5 * abs(det)
                areas[i, j, k] = area
    return areas

def update_areas(points, areas, idx):
    other_indices = [i for i in range(N) if i != idx]
    for a in range(len(other_indices)):
        for b in range(a+1, len(other_indices)):
            i = other_indices[a]
            j = other_indices[b]
            p1, p2, p3 = points[idx], points[i], points[j]
            v1 = p2 - p1
            v2 = p3 - p1
            det = v1[0]*v2[1] - v1[1]*v2[0]
            area = 0.5 * abs(det)
            indices = sorted([idx, i, j])
            areas[indices[0], indices[1], indices[2]] = area

def get_min_area(areas):
    min_area = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                if areas[i, j, k] < min_area:
                    min_area = areas[i, j, k]
    return min_area

def generate_lattice_points(M):
    points = []
    u = np.array([1.0, 0.0])
    v = np.array([0.5, np.sqrt(3)/2])
    for i in range(M+1):
        for j in range(M+1):
            p = (i*u + j*v) / M
            if is_inside_triangle(p):
                points.append(p)
    return np.array(points)

def initial_points(seed):
    rng = np.random.RandomState(seed)
    best_pts = None
    best_score = -1
    for M in [4, 5, 6]:
        candidates = generate_lattice_points(M)
        if len(candidates) < N:
            continue
        pts = []
        center = np.mean(candidates, axis=0)
        dists = np.linalg.norm(candidates - center, axis=1)
        idx = np.argmin(dists)
        pts.append(candidates[idx])
        
        for _ in range(N-1):
            best_idx = -1
            best_min_dist = -1
            for i, p in enumerate(candidates):
                if any(np.allclose(p, q) for q in pts):
                    continue
                min_dist = min(np.linalg.norm(p - q) for q in pts)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = i
            if best_idx >= 0:
                pts.append(candidates[best_idx])
            else:
                break
        
        if len(pts) == N:
            pts_arr = np.array(pts)
            # Add small random noise to break symmetry
            pts_arr += rng.uniform(-0.01, 0.01, size=(N, 2))
            # Project back to triangle if noise pushed out
            for i in range(N):
                if not is_inside_triangle(pts_arr[i]):
                    # Simple rejection
                    while True:
                        u, v = rng.random(), rng.random()
                        if u + v < 1:
                            pts_arr[i] = u * np.array([1,0]) + v * np.array([0.5, np.sqrt(3)/2])
                            break
            areas = init_areas(pts_arr)
            score = get_min_area(areas)
            if score > best_score:
                best_score = score
                best_pts = pts_arr.copy()
                
    if best_pts is None:
        best_pts = np.random.random((N, 2))
        for i in range(N):
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 1:
                    best_pts[i] = u * np.array([1,0]) + v * np.array([0.5, np.sqrt(3)/2])
                    break
                    
    return best_pts

def local_search(pts, time_limit, seed):
    start_time = time.time()
    areas = init_areas(pts)
    current_min_area = get_min_area(areas)
    rng = np.random.RandomState(seed)
    step_size = 0.05
    
    while time.time() < time_limit:
        idx = rng.randint(0, N)
        orig = pts[idx].copy()
        
        angle = rng.uniform(0, 2*np.pi)
        mag = step_size * rng.uniform(0.5, 1.5)
        dx = mag * np.cos(angle)
        dy = mag * np.sin(angle)
        
        new_pt = orig + np.array([dx, dy])
        
        if is_inside_triangle(new_pt):
            pts[idx] = new_pt
            update_areas(pts, areas, idx)
            new_min_area = get_min_area(areas)
            if new_min_area > current_min_area:
                current_min_area = new_min_area
            else:
                pts[idx] = orig
                update_areas(pts, areas, idx)
        
        if rng.rand() < 0.005:
            step_size *= 0.9
            if step_size < 0.001:
                step_size = 0.001
                
    return pts, current_min_area

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_pts = None
    best_score = -1
    
    # Multiple restarts
    num_restarts = 5
    time_per_restart = (deadline - start_time) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        pts = initial_points(i)
        pts, score = local_search(pts, deadline, i)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
            
    if best_pts is None:
        best_pts = initial_points(0)
        
    return best_pts

