import numpy as np
import time
from scipy.optimize import minimize

N = 11
H = np.sqrt(3) / 2.0
SQRT3 = np.sqrt(3)

# Precompute triangle indices
i_idx, j_idx, k_idx = [], [], []
for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            i_idx.append(i)
            j_idx.append(j)
            k_idx.append(k)
i_idx = np.array(i_idx)
j_idx = np.array(j_idx)
k_idx = np.array(k_idx)

def get_min_area(points):
    # points: (N, 2)
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(dx1 * dy2 - dx2 * dy1)
    return np.min(areas)

def boundary_violation(points):
    x = points[:, 0]
    y = points[:, 1]
    
    # y >= 0
    v1 = np.clip(-y, 0, None)
    # y <= SQRT3 * x
    v2 = np.clip(SQRT3 * x - y, 0, None)
    # y <= SQRT3 * (1-x)
    v3 = np.clip(SQRT3 * (1 - x) - y, 0, None)
    
    return np.sum(v1**2) + np.sum(v2**2) + np.sum(v3**2)

def objective(x):
    points = x.reshape(N, 2)
    pen = boundary_violation(points)
    area = get_min_area(points)
    # We want to maximize area, so minimize -area
    # Add penalty for going out of bounds
    return -area + 1000.0 * pen

def generate_greedy_start(rng, n_candidates=500):
    # Generate candidates in the triangle
    # Use rejection sampling
    candidates = []
    while len(candidates) < n_candidates:
        u, v = rng.random(), rng.random()
        if u + v < 1.0:
            x = u + 0.5 * v
            y = H * v
            candidates.append([x, y])
    candidates = np.array(candidates)
    
    selected = []
    # Pick first point randomly
    first_idx = rng.integers(n_candidates)
    selected.append(candidates[first_idx])
    
    for _ in range(N - 1):
        # Calculate min distance from each candidate to the selected set
        # selected is list of arrays
        sel_arr = np.array(selected)
        # dists shape: (n_candidates, len(selected))
        dists = np.linalg.norm(candidates[:, np.newaxis, :] - sel_arr[np.newaxis, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        
        max_idx = np.argmax(min_dists)
        selected.append(candidates[max_idx])
        
    return np.array(selected).flatten()

def generate_random_start(rng):
    pts = []
    while len(pts) < N:
        u, v = rng.random(), rng.random()
        if u + v < 1.0:
            x = u + 0.5 * v
            y = H * v
            pts.append([x, y])
    return np.array(pts).flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 105.0
    
    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Generate starts
    starts = []
    for _ in range(30):
        starts.append(generate_greedy_start(rng))
    for _ in range(5):
        starts.append(generate_random_start(rng))
        
    for s in starts:
        if time.time() > deadline:
            break
            
        try:
            # Nelder-Mead
            res = minimize(objective, s, method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            points = res.x.reshape(N, 2)
            
            # Check if valid
            viol = boundary_violation(points)
            if viol < 1e-4:
                score = get_min_area(points)
                if score > best_score:
                    best_score = score
                    best_points = points
        except Exception:
            continue
            
    if best_points is None:
        # Fallback
        return starts[0].reshape(N, 2)
        
    return best_points
