import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_positions = None
    best_score = -1.0
    
    def get_score(pos):
        # pos shape (N, 3)
        # Check bounds
        # min_i min_k (0.5 - |pos[i,k]|)
        wall_dists = 0.5 - np.abs(pos)
        min_wall = np.min(wall_dists)
        
        # Pairwise distances
        # Efficient calculation of min pairwise distance
        # N=50, so 50*49/2 = 1225 pairs. Vectorized is fast.
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        r = min(min_wall, min_pair / 2.0)
        return r

    def objective(pos_flat):
        pos = pos_flat.reshape(N, 3)
        # Add penalty if out of bounds? 
        # Bounds are handled by minimize, but just in case, 
        # if a point is out, the score calculation might be weird.
        # However, minimize with bounds keeps them in.
        return -get_score(pos)

    # Initial strategies
    def random_start():
        return np.random.uniform(-0.5, 0.5, (N, 3))

    def jittered_grid_start():
        # 4x4x4 grid
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        # We have 64 points. We need 50.
        # Randomly select 50 indices
        idx = np.random.choice(64, N, replace=False)
        pos = grid[idx]
        # Add small jitter to break symmetry
        pos += np.random.normal(0, 0.01, pos.shape)
        # Clip to bounds
        pos = np.clip(pos, -0.5, 0.5)
        return pos

    # Try a few different initializations
    initializations = []
    # 1. Random
    initializations.append(random_start())
    # 2. Jittered Grid
    initializations.append(jittered_grid_start())
    # 3. Another Random
    initializations.append(random_start())
    # 4. Another Jittered Grid
    initializations.append(jittered_grid_start())
    
    # Add more if time permits
    # We will iterate through these and also generate new random ones in a loop

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Powell method is derivative-free and good for this
    options = {
        'maxiter': 500,
        'maxfev': 5000
    }
    
    # Run optimization for each initial guess
    # We'll do a loop to allow for more restarts if time is available
    
    restarts_done = 0
    max_restarts = 50
    
    # First pass: the prepared initializations
    for init_pos in initializations:
        if time.time() > deadline:
            break
            
        res = minimize(objective, init_pos.flatten(), method='Powell', bounds=bounds, options=options)
        
        if res.success or res.fun < objective(init_pos.flatten()):
            curr_score = -res.fun
            if curr_score > best_score:
                best_score = curr_score
                best_positions = res.x.reshape(N, 3)
        
        restarts_done += 1
        
    # Second pass: Random restarts until time runs out
    while time.time() < deadline and restarts_done < max_restarts:
        init_pos = random_start()
        # We can reduce maxfev for later restarts to try more of them
        current_options = {'maxiter': 200, 'maxfev': 2000}
        
        res = minimize(objective, init_pos.flatten(), method='Powell', bounds=bounds, options=current_options)
        
        if res.success or res.fun < objective(init_pos.flatten()):
            curr_score = -res.fun
            if curr_score > best_score:
                best_score = curr_score
                best_positions = res.x.reshape(N, 3)
        
        restarts_done += 1

    if best_positions is None:
        # Fallback to seed-like grid if optimization failed completely
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_positions = grid[:N]
    else:
        # Final validation/clipping to ensure strict validity
        best_positions = np.clip(best_positions, -0.5, 0.5)

    return best_positions
