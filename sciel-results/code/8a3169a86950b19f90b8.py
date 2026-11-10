import numpy as np
import time

N = 100

def _compute_radius(centers):
    """Compute the common radius for a set of centers."""
    wall_clear = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_clear)
    
    if len(centers) > 1:
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
    else:
        min_pair = min_wall
    
    return min(min_pair, min_wall)

def _local_search(c, time_limit, rng):
    """
    Fast local search using coordinate ascent / hill climbing.
    Tries to move each sphere in small steps to increase the minimum clearance.
    """
    c = c.copy()
    current_r = _compute_radius(c)
    step = 0.01
    no_improve_count = 0
    start_time = time.time()
    
    # We will iterate until time runs out or we stagnate
    while time.time() - start_time < time_limit:
        improved = False
        
        # Iterate through all spheres, or a random subset if it takes too long
        # For N=100, iterating all is fast enough if we are careful.
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            
            # Try moving sphere i in 3D
            # We try to move it away from its closest neighbors and walls
            # Simple strategy: try random directions with current step
            # More efficient: calculate gradient of min-distance? 
            # For speed, we use random perturbations but with a bias towards "unblocking"
            
            # Let's try a few random directions
            for _ in range(5):
                direction = rng.normal(size=3)
                direction /= (np.linalg.norm(direction) + 1e-15)
                
                for sign in [1, -1]:
                    c_new = c.copy()
                    c_new[i] += sign * step * direction
                    # Clip to stay inside bounds (with small margin to avoid numerical issues)
                    c_new = np.clip(c_new, -0.4999, 0.4999)
                    
                    r_new = _compute_radius(c_new)
                    if r_new > current_r + 1e-12:
                        c = c_new
                        current_r = r_new
                        improved = True
                        no_improve_count = 0
                        break
                
                if improved:
                    break
        
        if not improved:
            no_improve_count += 1
            # Reduce step size
            step *= 0.8
            if step < 1e-6:
                break
            
            # If we've been stuck for a while, try a larger "shake" to escape local minima
            if no_improve_count > 20:
                # Randomly perturb a few spheres significantly
                for _ in range(5):
                    i = rng.randint(N)
                    c[i] += rng.uniform(-0.05, 0.05, size=3)
                    c[i] = np.clip(c[i], -0.4999, 0.4999)
                # Recalculate radius after shake
                current_r = _compute_radius(c)
                # Reset step to something moderate
                step = 0.01
                no_improve_count = 0
    
    return c, current_r

def _make_initial_guess(seed, lattice_type='cubic'):
    rng = np.random.RandomState(seed)
    
    if lattice_type == 'cubic':
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        grid = grid[:N]
        jitter = rng.uniform(-0.02, 0.02, size=(N, 3))
        grid = grid + jitter
        grid = np.clip(grid, -0.49, 0.49)
        return grid
    
    elif lattice_type == 'fcc':
        # Generate FCC lattice points
        points = []
        s = 0.38 # Approximate spacing
        # We need 100 points. A 5x5x5 cubic grid has 125 points.
        # FCC has higher density.
        # Let's generate a larger grid and take the closest 100 to center? 
        # Or just generate a dense FCC and clip.
        for i in range(-3, 4):
            for j in range(-3, 4):
                for k in range(-3, 4):
                    # Corners
                    points.append([i*s, j*s, k*s])
                    # Face centers
                    if i < 3: points.append([(i+0.5)*s, j*s, k*s])
                    if j < 3: points.append([i*s, (j+0.5)*s, k*s])
                    if k < 3: points.append([i*s, j*s, (k+0.5)*s])
        
        points = np.array(points)
        mask = np.all(np.abs(points) < 0.49, axis=1)
        points = points[mask]
        
        # If we have too many, select the ones closest to the center? 
        # Or just take the first N? Taking first N might be biased.
        # Let's sort by distance to center and take the first N? 
        # No, we want them spread out. 
        # Actually, for initialization, just taking a subset is fine, the local search will fix it.
        if len(points) >= N:
            # Randomly select N points to avoid bias
            indices = rng.choice(len(points), N, replace=False)
            points = points[indices]
        else:
            # Fill with random
            extra = rng.uniform(-0.49, 0.49, size=(N - len(points), 3))
            points = np.vstack([points, extra])
            
        jitter = rng.uniform(-0.005, 0.005, size=(N, 3))
        points = points + jitter
        points = np.clip(points, -0.49, 0.49)
        return points
    
    elif lattice_type == 'bcc':
        points = []
        s = 0.38
        for i in range(-3, 4):
            for j in range(-3, 4):
                for k in range(-3, 4):
                    points.append([i*s, j*s, k*s])
                    if i < 3 and j < 3 and k < 3:
                        points.append([(i+0.5)*s, (j+0.5)*s, (k+0.5)*s])
        points = np.array(points)
        mask = np.all(np.abs(points) < 0.49, axis=1)
        points = points[mask]
        if len(points) >= N:
            indices = rng.choice(len(points), N, replace=False)
            points = points[indices]
        else:
            extra = rng.uniform(-0.49, 0.49, size=(N - len(points), 3))
            points = np.vstack([points, extra])
        jitter = rng.uniform(-0.005, 0.005, size=(N, 3))
        points = points + jitter
        points = np.clip(points, -0.49, 0.49)
        return points
    
    else: # random
        points = rng.uniform(-0.49, 0.49, size=(N, 3))
        return points

def construct():
    start_time = time.time()
    # Use a safety margin. The limit is 120s. We will stop at 115s to be safe.
    time_limit = 115.0
    
    best_r = 0.0
    best_c = None
    rng = np.random.RandomState(42)
    
    # Strategy: 
    # 1. Try a few good initializations (Cubic, FCC, BCC, Random)
    # 2. Run local search on each for a portion of the time
    # 3. Keep the best result
    
    configs = [
        (0, 'cubic'),
        (1, 'fcc'),
        (2, 'bcc'),
        (3, 'cubic'),
        (4, 'fcc'),
        (5, 'random'),
    ]
    
    # We will allocate time to each config
    # If we have 115s, and 6 configs, we can give ~15s to each?
    # Or we can run them in parallel? No, single thread.
    # Let's run them sequentially with a time budget for each.
    
    # Actually, it's better to run one for a short time, then another, and keep the best.
    # Let's give each config a budget of 15 seconds.
    # 6 * 15 = 90s. Leaves 25s for a final polish on the best one.
    
    budget_per_config = 15.0
    
    for seed, lattice_type in configs:
        if time.time() - start_time > time_limit - 10: # Leave 10s for final polish
            break
            
        c0 = _make_initial_guess(seed, lattice_type)
        # Run local search
        c_res, r_res = _local_search(c0, budget_per_config, rng)
        
        if r_res > best_r:
            best_r = r_res
            best_c = c_res.copy()
            
    # Final polish on the best solution with remaining time
    if best_c is not None:
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 2:
            # Use a smaller step size for final polish? 
            # The _local_search function handles step size reduction automatically.
            # We just give it the remaining time.
            c_polished, r_polished = _local_search(best_c, remaining_time - 1, rng)
            if r_polished > best_r:
                best_r = r_polished
                best_c = c_polished.copy()
    else:
        # Fallback
        best_c = _make_initial_guess(0, 'cubic')
        
    return best_c
