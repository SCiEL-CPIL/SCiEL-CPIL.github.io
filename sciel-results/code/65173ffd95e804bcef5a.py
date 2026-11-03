"""Improved program using local search (hill climbing) and gradient polish."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _calculate_true_radius(centers):
    """Calculate the true maximum radius for a given set of centers."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _local_search(centers, max_iter=10000, time_limit=10.0):
    """
    Perform local search (hill climbing) to improve the packing.
    Tries to move each circle to a position that increases the minimum distance
    to its neighbors and walls.
    """
    centers = centers.copy()
    best_r = _calculate_true_radius(centers)
    step_size = 0.01
    no_improve_count = 0
    start_time = time.time()
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        
        indices = np.random.permutation(N)
        
        for idx in indices:
            orig_center = centers[idx].copy()
            
            wall_dist = 0.5 - np.abs(orig_center)
            current_min_wall = np.min(wall_dist)
            
            diffs = centers - orig_center
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            dists[idx] = np.inf
            current_min_pair = np.min(dists) / 2.0
            
            current_local_min = min(current_min_wall, current_min_pair)
            
            closest_idx = np.argmin(dists)
            if dists[closest_idx] < current_min_wall * 2: 
                direction = orig_center - centers[closest_idx]
                norm = np.linalg.norm(direction)
                if norm > 1e-9:
                    direction /= norm
            else:
                direction = -orig_center
                norm = np.linalg.norm(direction)
                if norm > 1e-9:
                    direction /= norm
                else:
                    direction = np.random.normal(0, 1, 2)
                    direction /= np.linalg.norm(direction)
            
            best_local_min = current_local_min
            best_new_center = orig_center
            
            n_trials = 80
            for _ in range(n_trials): 
                noise = np.random.normal(0, 1, 2)
                move = direction * step_size + noise * step_size * 0.5
                
                new_center = orig_center + move
                new_center = np.clip(new_center, -0.5, 0.5)
                
                new_wall_dist = 0.5 - np.abs(new_center)
                new_min_wall = np.min(new_wall_dist)
                
                new_diffs = centers - new_center
                new_dists = np.sqrt(np.sum(new_diffs**2, axis=1))
                new_dists[idx] = np.inf
                new_min_pair = np.min(new_dists) / 2.0
                
                new_local_min = min(new_min_wall, new_min_pair)
                
                if new_local_min > best_local_min:
                    best_local_min = new_local_min
                    best_new_center = new_center
            
            if best_local_min > current_local_min + 1e-12:
                centers[idx] = best_new_center
                improved = True
        
        if not improved:
            no_improve_count += 1
            step_size *= 0.85
            if no_improve_count > 80:
                break
        else:
            no_improve_count = 0
            
        current_r = _calculate_true_radius(centers)
        if current_r > best_r:
            best_r = current_r
            
    return centers, best_r

def _optimize_from_start(start_centers, bounds):
    """
    Hybrid optimization: Local Search followed by L-BFGS-B polish with multiple k values.
    """
    best_centers = start_centers.copy()
    best_score = _calculate_true_radius(best_centers)
    
    # Phase 1: Local Search
    ls_centers, ls_score = _local_search(best_centers, max_iter=8000, time_limit=25.0)
    if ls_score > best_score:
        best_score = ls_score
        best_centers = ls_centers
        
    # Phase 2: Gradient Polish with multiple k values
    def objective(params, k):
        centers = params.reshape(N, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_terms = dists / 2.0
        wall_dists = 0.5 - np.abs(centers)
        all_terms = np.concatenate([pair_terms.flatten(), wall_dists.flatten()])
        all_terms = all_terms[np.isfinite(all_terms)]
        if len(all_terms) == 0:
            return 0.0
        m = np.min(all_terms)
        shifted = all_terms - m
        exp_terms = np.exp(-k * shifted)
        log_sum = np.log(np.sum(exp_terms))
        soft_min = m - (1.0 / k) * log_sum
        return -soft_min

    for k_val in [500, 1000, 2000]:
        try:
            result = minimize(
                objective, 
                best_centers.reshape(-1), 
                args=(k_val,),
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            centers = result.x.reshape(N, 2)
            true_r = _calculate_true_radius(centers)
            
            if true_r > best_score:
                best_score = true_r
                best_centers = centers
        except Exception:
            pass
        
    # Phase 3: Final local search polish
    final_centers, final_score = _local_search(best_centers, max_iter=3000, time_limit=10.0)
    if final_score > best_score:
        best_score = final_score
        best_centers = final_centers
        
    return best_centers, best_score

def construct():
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    inits = []
    
    # 1. Hexagonal grid (tighter)
    s = 0.135
    points = []
    range_x = np.arange(-0.75, 0.75, s)
    range_y = np.arange(-0.75, 0.75, s)
    for i, x in enumerate(range_x):
        offset = (s / 2) if i % 2 == 0 else 0
        for y in range_y:
            points.append((x + offset, y))
    points = np.array(points)
    mask = (np.abs(points[:, 0]) <= 0.49) & (np.abs(points[:, 1]) <= 0.49)
    points = points[mask]
    if len(points) >= N:
        np.random.seed(42)
        indices = np.random.choice(len(points), N, replace=False)
        inits.append(points[indices])
        
    # 2. Square grid 7x7
    grid_x = np.linspace(-0.45, 0.45, 7)
    grid_y = np.linspace(-0.45, 0.45, 7)
    grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
    if len(grid_points) >= N:
        np.random.seed(123)
        indices = np.random.choice(len(grid_points), N, replace=False)
        inits.append(grid_points[indices])
        
    # 3. Random uniform
    np.random.seed(999)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 4. Jittered hex
    if len(points) >= N:
        np.random.seed(43)
        indices = np.random.choice(len(points), N, replace=False)
        jittered = points[indices] + np.random.normal(0, 0.015, (N, 2))
        jittered = np.clip(jittered, -0.49, 0.49)
        inits.append(jittered)
        
    # 5. Another random
    np.random.seed(1000)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 6. Randomized Grid
    np.random.seed(555)
    grid_x_rand = np.sort(np.random.uniform(-0.49, 0.49, 7))
    grid_y_rand = np.sort(np.random.uniform(-0.49, 0.49, 7))
    grid_points_rand = np.array([[x, y] for x in grid_x_rand for y in grid_y_rand])
    if len(grid_points_rand) >= N:
        indices = np.random.choice(len(grid_points_rand), N, replace=False)
        inits.append(grid_points_rand[indices])
        
    # 7. 10x5 rectangular grid
    grid_x10 = np.linspace(-0.45, 0.45, 10)
    grid_y5 = np.linspace(-0.45, 0.45, 5)
    grid_points105 = np.array([[x, y] for x in grid_x10 for y in grid_y5])
    if len(grid_points105) >= N:
        np.random.seed(777)
        indices = np.random.choice(len(grid_points105), N, replace=False)
        inits.append(grid_points105[indices])
        
    # 8. 5x10 rectangular grid
    grid_x5 = np.linspace(-0.45, 0.45, 5)
    grid_y10 = np.linspace(-0.45, 0.45, 10)
    grid_points510 = np.array([[x, y] for x in grid_x5 for y in grid_y10])
    if len(grid_points510) >= N:
        np.random.seed(888)
        indices = np.random.choice(len(grid_points510), N, replace=False)
        inits.append(grid_points510[indices])

    best_centers = None
    best_score = -1.0
    
    start_time = time.time()
    
    for i, init in enumerate(inits):
        if time.time() - start_time > 105.0: 
            break
            
        centers, score = _optimize_from_start(init, bounds)
        if score > best_score:
            best_score = score
            best_centers = centers
            
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
