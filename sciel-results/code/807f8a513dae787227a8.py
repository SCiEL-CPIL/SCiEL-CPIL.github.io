import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    configs = [
        [5,4,5,4,3],
        [5,4,4,4,4],
        [4,5,4,5,3],
        [3,5,4,5,4],
        [4,5,3,5,4],
        [5,3,5,4,4],
        [4,4,5,4,4],
    ]
    
    best_result = None
    best_sum = 0
    
    for config in configs:
        if sum(config) != N:
            continue
        if time.time() > deadline:
            break
        
        result, total_sum = _hex_packing(config)
        if result is not None and total_sum > best_sum:
            best_sum = total_sum
            best_result = result
    
    if best_result is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result

def _hex_packing(config):
    r = 1.0
    sqrt3 = np.sqrt(3)
    
    points = []
    for row_idx, n_circles in enumerate(config):
        y = r + row_idx * r * sqrt3
        if row_idx % 2 == 0:
            x_start = r
        else:
            x_start = 2*r
        
        for i in range(n_circles):
            x = x_start + 2*r*i
            points.append((x, y))
    
    if len(points) != N:
        return None, 0
    
    points = np.array(points)
    
    x_min = points[:, 0].min()
    x_max = points[:, 0].max()
    y_min = points[:, 1].min()
    y_max = points[:, 1].max()
    
    W = x_max - x_min + 2*r
    H = y_max - y_min + 2*r
    
    s = 2.0 / (W + H)
    r_scaled = r * s
    
    cx = (points[:, 0] - x_min) * s + r_scaled
    cy = (points[:, 1] - y_min) * s + r_scaled
    
    result = np.column_stack([cx, cy, np.full(N, r_scaled)])
    
    if _is_valid(result):
        return result, result[:, 2].sum()
    else:
        return None, 0

def _is_valid(result):
    if result is None or len(result) != N:
        return False
    
    x_min = (result[:, 0] - result[:, 2]).min()
    x_max = (result[:, 0] + result[:, 2]).max()
    y_min = (result[:, 1] - result[:, 2]).min()
    y_max = (result[:, 1] + result[:, 2]).max()
    
    W = x_max - x_min
    H = y_max - y_min
    
    if 2*(W + H) > 4 + 1e-9:
        return False
    
    for i in range(N):
        for j in range(i+1, N):
            dist_sq = (result[i,0]-result[j,0])**2 + (result[i,1]-result[j,1])**2
            min_dist = result[i,2] + result[j,2]
            if dist_sq < min_dist*min_dist - 1e-12:
                return False
    
    return True
