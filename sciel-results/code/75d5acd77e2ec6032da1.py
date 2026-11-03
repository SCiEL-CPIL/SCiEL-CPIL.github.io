"""Pack 50 equal circles in a unit square, maximize common radius."""
import numpy as np
import time

N = 50
DEADLINE = 135.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers."""
    # Distance to walls
    # Square is [-0.5, 0.5]^2
    wall_dist = np.min(np.abs(centers - 0.5), axis=1)
    # Wall radius = min of (0.5 - |coord|)
    wall_r = np.min(0.5 - np.abs(centers))
    # Pairwise distances
    # Use broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # Set diagonal to inf
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _objective(x):
    """Negative radius for minimization."""
    centers = x.reshape(N, 2)
    r = _radius(centers)
    return -r

def _make_grid(n_per_side, N):
    """Create a grid of N points in [-0.5, 0.5]^2."""
    # Use n_per_side x n_per_side grid, take first N
    ticks = (np.arange(n_per_side) + 0.5) / n_per_side - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _jitter(centers, scale):
    """Add random jitter to break symmetries."""
    return centers + np.random.normal(0, scale, size=centers.shape)

def _local_refine(centers, iters, step_size, time_limit):
    """Simple iterative local optimization: move points to maximize min radius."""
    centers = centers.copy()
    best_r = _radius(centers)
    t0 = time.time()
    
    for it in range(iters):
        if time.time() - t0 > time_limit:
            break
        
        r = best_r
        # Try to increase radius by moving points
        for i in range(N):
            if time.time() - t0 > time_limit:
                break
            # Find constraint direction
            # Compute gradient-like direction
            # Simple: try random perturbations and keep best
            best_i = centers[i].copy()
            best_r_i = r
            
            # Try a few random directions
            for _ in range(5):
                trial = centers.copy()
                trial[i] = best_i + np.random.normal(0, step_size, size=2)
                # Keep in bounds
                trial[i] = np.clip(trial[i], -0.5 + 0.01, 0.5 - 0.01)
                r_trial = _radius(trial)
                if r_trial > best_r_i:
                    best_r_i = r_trial
                    best_i = trial[i]
            
            if best_r_i > r:
                centers[i] = best_i
                r = best_r_i
        
        if r > best_r:
            best_r = r
    
    return centers, best_r

def construct():
    global DEADLINE
    start_time = time.time()
    
    # Start with a 7x7 grid (49 points) + 1 center
    # Actually, use 7x7 = 49, add 50th at center, but that's a duplicate.
    # Better: use a 7x7 grid but shift one point or use 7x7 with 50th inserted.
    # Let's use a 7x7 grid and place the 50th point by optimizing its position.
    
    # Initial: 7x7 grid
    centers = _make_grid(7, 50)  # This takes first 50 of 49x49... no, 7x7=49
    
    # Fix: _make_grid(7, 50) returns 50 points from a 7x7 grid (49 points) + need 1 more
    # Let me fix _make_grid to handle N > n_per_side^2
    
    # Actually, let's use a different approach:
    # Start with 7x7 grid (49 points), then add 1 point at center (0,0)
    # But (0,0) is already in the grid for 7x7? Let's check.
    # 7x7 grid: ticks = (0,1,2,3,4,5,6)/7 - 0.5 = -0.5, -4/7, -3/7, -2/7, -1/7, 0, 1/7, 2/7, 3/7, 4/7, 0.5
    # Wait, (np.arange(7) + 0.5)/7 - 0.5 = [0.5/7 - 0.5, 1.5/7 - 0.5, ...] 
    # = [-3/7, -2/7, -1/7, 0, 1/7, 2/7, 3/7]
    # So (0,0) is at index (3,3). We have a duplicate if we add another.
    
    # Better approach: use 7x7 grid and perturb the 50th point away.
    # Or use a 7x7 grid and optimize all 50 points.
    
    # Let's use a different initialization: 7x7 grid with 50th point placed
    # by finding the best empty spot.
    
    centers_49 = _make_grid(7, 49)
    # Add 50th point at (0,0) but offset slightly
    centers = np.vstack([centers_49, np.array([0.0, 0.0])])
    # Now we have 50 points with one duplicate. Jitter all.
    centers = _jitter(centers, 0.01)
    
    best_centers = centers.copy()
    best_r = _radius(centers)
    
    # Try a few different initial configurations
    configs = []
    configs.append(centers.copy())
    
    # Try 8x8 grid with 50 points
    centers_8x8 = _make_grid(8, 50)
    centers_8x8 = _jitter(centers_8x8, 0.01)
    configs.append(centers_8x8)
    
    # Try random initial points
    for seed in range(5):
        np.random.seed(seed)
        centers_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        configs.append(centers_rand)
    
    # Local refine each config
    for i, cfg in enumerate(configs):
        if time.time() - start_time > DEADLINE - 30:
            break
        time_left = DEADLINE - 30 - (time.time() - start_time)
        if time_left <= 0:
            break
        refined, r = _local_refine(cfg, iters=100, step_size=0.02, time_limit=time_left / len(configs))
        if r > best_r:
            best_r = r
            best_centers = refined
    
    # Global optimization if time remains
    if time.time() - start_time < DEADLINE - 10:
        time_left = DEADLINE - 10 - (time.time() - start_time)
        try:
            from scipy.optimize import differential_evolution
            # Use differential evolution on the best configuration
            # This is expensive, so use small population
            x0 = best_centers.ravel()
            bounds = [(-0.5, 0.5)] * (2 * N)
            result = differential_evolution(
                _objective, bounds,
                maxiter=50, tol=1e-6,
                popsize=15, seed=42,
                update_interval=1,
                polish=False
            )
            if result.fun < -best_r:
                best_centers = result.x.reshape(N, 2)
                best_r = -result.fun
        except Exception:
            pass
    
    return best_centers
