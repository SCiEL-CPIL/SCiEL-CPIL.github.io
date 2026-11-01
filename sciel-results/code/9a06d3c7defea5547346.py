import numpy as np
from scipy.optimize import minimize
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 110.0  # 10s safety margin

def score(centers):
    if np.any(np.isnan(centers)) or np.any(np.isinf(centers)):
        return -np.inf
    if np.any(np.abs(centers) > 0.5):
        return -np.inf
        
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    wall_dist = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_dist)
    
    return min(min_dist / 2.0, min_wall)

def objective(centers_flat):
    centers = centers_flat.reshape(N, 3)
    if np.any(np.abs(centers) > 0.5):
        # Strong penalty for out-of-bounds
        return 1.0 + np.sum((np.abs(centers) - 0.5)**2)
    r = score(centers)
    return -r

def construct():
    best_centers = None
    best_score = -np.inf
    
    # Strategy 1: FCC Greedy Init + Nelder-Mead
    # Generate FCC points
    a = 0.1
    coords = np.arange(-0.5, 0.5 + a/2, a)
    points = []
    for x in coords:
        for y in coords:
            for z in coords:
                points.append([x, y, z])
                if x + a/2 <= 0.5 and y + a/2 <= 0.5: points.append([x + a/2, y + a/2, z])
                if x + a/2 <= 0.5 and z + a/2 <= 0.5: points.append([x + a/2, y, z + a/2])
                if y + a/2 <= 0.5 and z + a/2 <= 0.5: points.append([x, y + a/2, z + a/2])
                
    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.5, axis=1)
    points = points[mask]
    
    if len(points) >= N:
        # Greedy selection
        # Start with point closest to center
        dists_to_center = np.sum(points**2, axis=1)
        start_idx = np.argmin(dists_to_center)
        
        selected = [points[start_idx]]
        remaining_indices = list(range(len(points)))
        remaining_indices.remove(start_idx)
        
        for _ in range(N - 1):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            # Vectorized min distance to selected set
            # selected is a list of arrays, convert to array for speed
            sel_arr = np.array(selected)
            # (M, 1, 3) - (1, M, 3) -> (M, M, 3)
            # Actually, we want dist from each remaining point (R, 3) to each selected (S, 3)
            # Result (R, S). Min over S.
            rem_arr = points[remaining_indices]
            d = np.linalg.norm(rem_arr[:, np.newaxis, :] - sel_arr[np.newaxis, :, :], axis=2)
            min_d = np.min(d, axis=1)
            best_local_idx = np.argmax(min_d)
            best_global_idx = remaining_indices[best_local_idx]
            selected.append(points[best_global_idx])
            remaining_indices.remove(best_global_idx)
            
        if len(selected) == N:
            centers = np.array(selected)
            s = score(centers)
            if s > best_score:
                best_score = s
                best_centers = centers
            
            # Optimize
            if time.time() - START_TIME < TIME_LIMIT - 10:
                x0 = centers.flatten()
                try:
                    res = minimize(objective, x0, method='Nelder-Mead', 
                                   options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-7})
                    opt_centers = res.x.reshape(N, 3)
                    opt_score = score(opt_centers)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_centers = opt_centers
                except Exception:
                    pass

    # Fallback: Grid if FCC failed or time ran out
    if best_centers is None:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    # Final validity check
    if np.any(np.abs(best_centers) > 0.5):
        best_centers = np.clip(best_centers, -0.5, 0.5)
        
    return best_centers
