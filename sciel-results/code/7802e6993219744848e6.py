"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    max_time = 115.0  # Safe limit
    
    def calculate_radii(pos):
        """
        pos: (N, 2) array of (x, y)
        Returns: (N,) array of max possible radii
        """
        # Wall distances
        # r_i <= x_i, 1-x_i, y_i, 1-y_i
        dist_walls = np.minimum(pos, 1.0 - pos)
        r = np.min(dist_walls, axis=1)
        
        # Pairwise distances
        # dist[i, j]
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Ignore self
        np.fill_diagonal(dists, np.inf)
        
        # r_i <= min_{j != i} dist(i, j) / 2
        min_pair_dist = np.min(dists, axis=1)
        r_pair = min_pair_dist / 2.0
        
        r = np.minimum(r, r_pair)
        return r
    
    def score(pos):
        return np.sum(calculate_radii(pos))
    
    best_pos = None
    best_score = -np.inf
    
    # Initialization: Hexagonal lattice
    def init_hex():
        # Try to fit a hex lattice
        # Spacing s
        # Area per point approx s^2 * sqrt(3)/2
        # Total area 1. N=26.
        # s^2 ~ 1/26 -> s ~ 0.2
        s = 0.2
        points = []
        # Generate grid
        for j in range(8):
            y = 0.1 + j * s * np.sqrt(3) / 2
            if y > 0.9: break
            for i in range(8):
                x = 0.1 + i * s + (j % 2) * s / 2
                if x > 0.9: break
                points.append((x, y))
        
        # If we have more than N, pick N randomly? Or the first N?
        # Better to pick N that are most "central" or just random subset
        if len(points) > N:
            # Shuffle and take N
            np.random.shuffle(points)
            points = points[:N]
        else:
            # If too few, pad with random
            while len(points) < N:
                points.append((np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)))
                
        pos = np.array(points)
        # Add small noise to break symmetry
        pos += np.random.normal(0, 0.01, pos.shape)
        return np.clip(pos, 0.05, 0.95)

    def init_grid():
        side = int(np.ceil(np.sqrt(N)))
        xs = np.linspace(0.5/side, 1 - 0.5/side, side)
        ys = np.linspace(0.5/side, 1 - 0.5/side, side)
        grid = np.array([(x, y) for x in xs for y in ys])
        np.random.shuffle(grid)
        return grid[:N] + np.random.normal(0, 0.01, (N, 2))

    def run_sa(pos_init, max_iters):
        pos = pos_init.copy()
        curr_score = score(pos)
        T = 0.02
        T_min = 0.0001
        cooling = 0.9995
        step_size = 0.05
        
        local_best_pos = pos.copy()
        local_best_score = curr_score
        
        for i in range(max_iters):
            if time.time() - start_time > max_time:
                break
                
            if T < T_min:
                break
                
            k = np.random.randint(N)
            angle = np.random.uniform(0, 2*np.pi)
            mag = np.random.uniform(0, step_size)
            
            new_pos = pos.copy()
            new_pos[k, 0] += mag * np.cos(angle)
            new_pos[k, 1] += mag * np.sin(angle)
            
            # Clip to bounds
            new_pos = np.clip(new_pos, 0, 1)
            
            new_score = score(new_pos)
            delta = new_score - curr_score
            
            if delta > 0:
                pos = new_pos
                curr_score = new_score
                if curr_score > local_best_score:
                    local_best_score = curr_score
                    local_best_pos = pos.copy()
            else:
                # Accept with probability
                if np.random.rand() < np.exp(-delta / T):
                    pos = new_pos
                    curr_score = new_score
            
            T *= cooling
            
        return local_best_pos, local_best_score

    # Run multiple restarts
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() - start_time > max_time:
            break
            
        if r % 2 == 0:
            pos_init = init_hex()
        else:
            pos_init = init_grid()
            
        # Increase iterations for earlier restarts?
        # Just use a fixed budget per restart
        max_iters = 20000
        pos_res, score_res = run_sa(pos_init, max_iters)
        
        if score_res > best_score:
            best_score = score_res
            best_pos = pos_res

    if best_pos is None:
        # Fallback
        pos_init = init_grid()
        best_pos = pos_init
        best_score = score(best_pos)

    radii = calculate_radii(best_pos)
    result = np.column_stack((best_pos, radii))
    return result

# EVOLVE-BLOCK-END
