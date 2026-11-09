import numpy as np
import time

N = 200

def construct():
    start_time = time.time()
    time_limit = 115.0  # 5s safety margin
    
    # 1. Initialization: Triangular Lattice
    s = 1.0 / 14.0
    h = s * np.sqrt(3) / 2
    
    points = []
    max_x = int(0.5 / s) + 1
    max_y = int(0.5 / h) + 1
    
    for j in range(-max_y, max_y + 1):
        y = j * h
        if y < -0.5 or y > 0.5:
            continue
        offset = 0.0 if j % 2 == 0 else s / 2.0
        for i in range(-max_x, max_x + 1):
            x = i * s + offset
            if x < -0.5 or x > 0.5:
                continue
            points.append([x, y])
            
    points = np.array(points)
    
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        diff = N - len(points)
        random_pts = np.random.uniform(-0.5, 0.5, (diff, 2))
        points = np.vstack([points, random_pts])
        
    best_points = points.copy()
    best_score = -1.0
    
    def get_score(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        min_boundary_dist = np.min([
            0.5 - np.min(pts[:, 0]),
            0.5 - np.max(pts[:, 0]),
            0.5 - np.min(pts[:, 1]),
            0.5 - np.max(pts[:, 1])
        ])
        
        return min(min_pair_dist / 2.0, min_boundary_dist)

    current_score = get_score(points)
    best_score = current_score
    
    step_size = 0.005
    jiggle_prob = 0.05
    
    # Optimization Loop
    while time.time() - start_time < time_limit:
        # Calculate distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_val = np.min(dists)
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        
        boundary_dists = np.array([
            0.5 - np.min(points[:, 0]),
            0.5 - np.max(points[:, 0]),
            0.5 - np.min(points[:, 1]),
            0.5 - np.max(points[:, 1])
        ])
        min_boundary = np.min(boundary_dists)
        
        if min_boundary < min_val / 2.0:
            # Boundary bottleneck
            if np.argmin(boundary_dists) == 0:
                idx = np.argmin(points[:, 0])
                points[idx, 0] += step_size
            elif np.argmin(boundary_dists) == 1:
                idx = np.argmax(points[:, 0])
                points[idx, 0] -= step_size
            elif np.argmin(boundary_dists) == 2:
                idx = np.argmin(points[:, 1])
                points[idx, 1] += step_size
            else:
                idx = np.argmax(points[:, 1])
                points[idx, 1] -= step_size
        else:
            # Pair bottleneck
            vec = points[i] - points[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec = vec / norm
            else:
                vec = np.array([1.0, 0.0])
            
            points[i] += vec * step_size
            points[j] -= vec * step_size
            
            if np.random.random() < jiggle_prob:
                k = np.random.randint(N)
                points[k] += np.random.normal(0, step_size, 2)
        
        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
        
        new_score = get_score(points)
        if new_score > best_score:
            best_score = new_score
            best_points = points.copy()
            
    return best_points
