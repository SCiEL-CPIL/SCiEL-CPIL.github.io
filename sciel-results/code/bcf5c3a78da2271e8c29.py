import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    start_time = time.time()
    deadline = start_time + 590

    # Precompute pair indices for pdist
    pairs = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            pairs.append((i, j))
    pair_indices = np.array(pairs)
    num_pairs = len(pairs)

    def get_score(points):
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10: return 0.0
        return (d_min / d_max) ** 2

    def nudge_local_search(points, max_iter=2000):
        pts = points.copy()
        best_score = get_score(pts)
        best_pts = pts.copy()
        
        for i in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            dists = pdist(pts, 'euclidean')
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max < 1e-10:
                break
            
            current_score = (d_min / d_max) ** 2
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
                
            min_idx = np.argmin(dists)
            max_idx = np.argmax(dists)
            
            i_min, j_min = pair_indices[min_idx]
            i_max, j_max = pair_indices[max_idx]
            
            new_pts = pts.copy()
            
            vec_min = pts[j_min] - pts[i_min]
            norm_min = np.linalg.norm(vec_min)
            if norm_min > 1e-10:
                unit_min = vec_min / norm_min
                step = 0.01 * norm_min
                new_pts[i_min] -= unit_min * step
                new_pts[j_min] += unit_min * step
            
            vec_max = pts[j_max] - pts[i_max]
            norm_max = np.linalg.norm(vec_max)
            if norm_max > 1e-10:
                unit_max = vec_max / norm_max
                step = 0.01 * norm_max
                new_pts[i_max] += unit_max * step
                new_pts[j_max] -= unit_max * step
                
            noise_scale = 0.05 * (d_max - d_min)
            if noise_scale > 1e-10:
                new_pts += np.random.normal(0, noise_scale, size=new_pts.shape)
                
            new_score = get_score(new_pts)
            if new_score > best_score:
                pts = new_pts
                best_score = new_score
                best_pts = new_pts.copy()
            elif new_score > 0 and np.random.rand() < 0.1:
                pts = new_pts
                
        return best_pts

    best_points = None
    best_score = -1.0

    def update_best(pts):
        nonlocal best_points, best_score
        if pts is None or not np.all(np.isfinite(pts)): return
        s = get_score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    initial_configs = []
    
    x = np.arange(4)
    y = np.arange(4)
    xx, yy = np.meshgrid(x, y)
    grid_pts = np.column_stack((xx.ravel(), yy.ravel()))
    initial_configs.append(grid_pts)
    
    hex_pts = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0],
        [0.5, 0.866], [1.5, 0.866], [2.5, 0.866], [3.5, 0.866],
        [0, 1.732], [1, 1.732], [2, 1.732], [3, 1.732],
        [0.5, 2.598], [1.5, 2.598], [2.5, 2.598], [3.5, 2.598]
    ])
    initial_configs.append(hex_pts)
    
    for _ in range(10):
        initial_configs.append(np.random.rand(n, 2))
        
    for init_pts in initial_configs:
        if time.time() > deadline - 30:
            break
        improved_pts = nudge_local_search(init_pts, max_iter=1000)
        update_best(improved_pts)
        
    if best_points is None:
        return initial_configs[0]
        
    return best_points
