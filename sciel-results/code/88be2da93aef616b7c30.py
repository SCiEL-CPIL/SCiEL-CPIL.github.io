import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s safety margin

    best_points = None
    best_score = -1.0

    def calc_score(pts):
        # Min distance to boundary
        dist_bound = 1.0 - np.linalg.norm(pts, axis=1)
        min_bound = np.min(dist_bound)
        
        # Min distance between pairs
        dist_matrix = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix) / 2.0
        
        return min(min_bound, min_pair)

    # Helper to generate hexagonal lattice points
    def get_hex_points(spacing):
        points = []
        max_idx = int(2.0 / spacing) + 2
        for j in range(-max_idx, max_idx + 1):
            for i in range(-max_idx, max_idx + 1):
                x = i * spacing + (j % 2) * spacing * 0.5
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y < 1.0:
                    points.append([x, y])
        return np.array(points)

    # Try a few different initializations
    spacings = [0.105, 0.11, 0.115]
    
    for s in spacings:
        if time.time() - start_time > time_limit:
            break
            
        lattice_points = get_hex_points(s)
        if len(lattice_points) < N:
            continue
        
        # Select N points closest to center
        dists = np.linalg.norm(lattice_points, axis=1)
        indices = np.argsort(dists)[:N]
        init_points = lattice_points[indices].copy()
        
        # Add small random noise to break symmetry
        init_points += np.random.randn(N, 2) * 0.001
        
        # Optimization Loop
        current_points = init_points
        step_size = 0.02
        decay = 0.9995
        local_best_score = calc_score(current_points)
        
        if local_best_score > best_score:
            best_score = local_best_score
            best_points = current_points.copy()
            
        for iter in range(3000):
            if time.time() - start_time > time_limit:
                break
                
            # Calculate distances
            dist_matrix = np.linalg.norm(current_points[:, np.newaxis, :] - current_points[np.newaxis, :, :], axis=2)
            np.fill_diagonal(dist_matrix, np.inf)
            
            min_dist_indices = np.argmin(dist_matrix, axis=1)
            min_dist_values = np.min(dist_matrix, axis=1)
            
            bound_dist = 1.0 - np.linalg.norm(current_points, axis=1)
            
            directions = np.zeros_like(current_points)
            
            for i in range(N):
                # Check if boundary is the active constraint
                if bound_dist[i] < min_dist_values[i] / 2.0:
                    norm_i = np.linalg.norm(current_points[i])
                    if norm_i > 1e-6:
                        directions[i] = -current_points[i] / norm_i
                    else:
                        directions[i] = np.random.randn(2)
                else:
                    j = min_dist_indices[i]
                    diff = current_points[i] - current_points[j]
                    norm_diff = np.linalg.norm(diff)
                    if norm_diff > 1e-6:
                        directions[i] = diff / norm_diff
                    else:
                        directions[i] = np.random.randn(2)
            
            current_points += directions * step_size
            
            # Project to unit circle
            norms = np.linalg.norm(current_points, axis=1)
            out_mask = norms > 1.0
            if np.any(out_mask):
                current_points[out_mask] /= norms[out_mask, np.newaxis]
            
            score = calc_score(current_points)
            if score > local_best_score:
                local_best_score = score
                if score > best_score:
                    best_score = score
                    best_points = current_points.copy()
            
            step_size *= decay
            if step_size < 1e-5:
                break

    return best_points
