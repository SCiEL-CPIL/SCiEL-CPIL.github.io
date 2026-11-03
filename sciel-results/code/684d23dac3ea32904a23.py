import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START
def construct():
    """
    Uses Simulated Annealing to maximize the minimum distance between circles
    and between circles and the square boundary.
    """
    
    def evaluate(points):
        """
        Returns the minimum radius r for the given configuration.
        r = min(min_pair_dist / 2, min_wall_dist)
        """
        # 1. Distance to walls
        wall_dists = np.minimum.reduce([
            0.5 - points[:, 0],
            0.5 + points[:, 0],
            0.5 - points[:, 1],
            0.5 + points[:, 1]
        ])
        min_wall = np.min(wall_dists)
        
        # 2. Distance between pairs
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        r = min(min_wall, min_pair / 2.0)
        return r

    def get_bottleneck(points):
        """
        Returns (bottleneck_type, indices, value) where bottleneck_type is 'wall' or 'pair'
        """
        wall_dists = np.minimum.reduce([
            0.5 - points[:, 0],
            0.5 + points[:, 0],
            0.5 - points[:, 1],
            0.5 + points[:, 1]
        ])
        min_wall_val = np.min(wall_dists)
        min_wall_idx = np.argmin(wall_dists)
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_val = np.min(dists)
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        
        if min_wall_val < min_pair_val / 2.0:
            return 'wall', (min_wall_idx,), min_wall_val
        else:
            return 'pair', min_pair_idx, min_pair_val / 2.0

    def improve(points, temp):
        """
        Perform a single step of optimization.
        Strategy:
        1. Find the bottleneck (min wall or min pair).
        2. Move the involved points to increase the distance.
        3. Add random noise scaled by temp.
        """
        p = points.copy()
        
        bottleneck_type, indices, bottleneck_val = get_bottleneck(p)
        
        # Adaptive step size: larger when temp is high, smaller when low
        step_size = 0.02 * temp / 0.05 + 0.005
        
        if bottleneck_type == 'wall':
            idx = indices[0]
            x, y = p[idx]
            # Determine direction to move: towards center
            dx_pos = 0.5 - x
            dx_neg = 0.5 + x
            dy_pos = 0.5 - y
            dy_neg = 0.5 + y
            
            min_dist = min(dx_pos, dx_neg, dy_pos, dy_neg)
            
            # Move towards center
            if min_dist == dx_pos:
                p[idx, 0] -= step_size
            elif min_dist == dx_neg:
                p[idx, 0] += step_size
            elif min_dist == dy_pos:
                p[idx, 1] -= step_size
            else:
                p[idx, 1] += step_size
            
            # Add random noise to escape local minima
            p[idx] += np.random.normal(0, temp * 0.5, size=2)
            
        else:
            i, j = indices
            vec = p[j] - p[i]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
            else:
                vec = np.array([1.0, 0.0])
            
            # Move i and j apart
            p[i] -= vec * step_size
            p[j] += vec * step_size
            
            # Add random noise
            p[i] += np.random.normal(0, temp * 0.5, size=2)
            p[j] += np.random.normal(0, temp * 0.5, size=2)
            
        # Also add small random perturbation to all points to explore
        p += np.random.normal(0, temp * 0.1, size=p.shape)
        
        # Clip all points to stay inside (safety)
        p = np.clip(p, -0.5, 0.5)
        
        return p

    # Initialization: Grid with jitter
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    points = grid[:N].copy()
    points += np.random.normal(0, 0.01, size=points.shape)
    points = np.clip(points, -0.49, 0.49)
    
    best_points = points.copy()
    best_score = evaluate(best_points)
    
    # Simulated Annealing parameters
    initial_temp = 0.05
    final_temp = 0.001
    cooling_rate = (final_temp / initial_temp) ** (1.0 / 10000)
    
    # Time limit check
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    current_points = points.copy()
    current_score = best_score
    temp = initial_temp
    
    # Main SA loop
    while time.time() - start_time < time_limit:
        new_points = improve(current_points, temp)
        new_score = evaluate(new_points)
        
        # Accept if better, or with probability exp((new - current) / temp)
        if new_score >= current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            prob = np.exp((new_score - current_score) / temp)
            if np.random.rand() < prob:
                current_points = new_points
                current_score = new_score
        
        temp *= cooling_rate
        
        # Reset temperature occasionally to escape local minima
        if temp < final_temp * 2:
            temp = initial_temp * 0.5
    
    return best_points
# EVOLVE-BLOCK-END
