"""Combined program using diverse initialization and scipy Nelder-Mead optimization."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
import time
from scipy.optimize import minimize

def _radius_for_centers(centers):
    """Compute the feasible radius for given centers inside [-0.5, 0.5]^3."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Wall distances: min over each coordinate of (0.5 - |c|)
    # We need the minimum distance to any wall for any sphere
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    
    if min_wall < 0:
        return 0.0
        
    # Pairwise distances
    # Compute all pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)

def _neg_radius(centers_flat):
    """Negative radius for minimization."""
    centers = centers_flat.reshape(N, 3)
    return -_radius_for_centers(centers)

def _make_grid_initial(seed, dims):
    """Create a grid-based initial configuration with jitter."""
    rng = np.random.RandomState(seed)
    m1, m2, m3 = dims
    ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
    grid = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
    
    # If grid has more points than N, take first N
    # If fewer, we might need to add random points, but for 50, 4x5x3=60 is enough
    if len(grid) >= N:
        centers = grid[:N].copy()
    else:
        # Pad with random points
        extra_needed = N - len(grid)
        extra = rng.uniform(-0.4, 0.4, size=(extra_needed, 3))
        centers = np.vstack([grid, extra])
        
    # Add jitter to break symmetry
    centers += rng.uniform(-0.02, 0.02, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    return centers

def _make_random_initial(seed):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 3))

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We should aim to finish by 140s.
    deadline = start_time + 140.0
    
    best_centers = None
    best_r = 0.0
    
    # Define several initial strategies
    # 1. 4x5x3 grid (60 points, take 50) - Good spatial coverage
    # 2. 5x4x3 grid
    # 3. 3x5x4 grid
    # 4. Random
    # 5. Random with larger spread
    
    initializations = [
        _make_grid_initial(42, (4, 5, 3)),
        _make_grid_initial(123, (5, 4, 3)),
        _make_grid_initial(456, (3, 5, 4)),
        _make_random_initial(7),
        _make_random_initial(8),
    ]
    
    # Optimize each initialization
    for i, init_centers in enumerate(initializations):
        if time.time() > deadline - 10.0:
            break
            
        current = init_centers.copy()
        current_r = _radius_for_centers(current)
        
        # Use Nelder-Mead to maximize radius (minimize negative radius)
        x0 = current.flatten()
        
        # We can run multiple optimization attempts or just one long one
        # Nelder-Mead is stochastic in its simplex updates, so restarting from 
        # slightly perturbed points can help.
        
        for attempt in range(3):
            if time.time() > deadline - 5.0:
                break
                
            try:
                # Perturb the starting point slightly for subsequent attempts
                if attempt > 0:
                    rng = np.random.RandomState(100 + i + attempt)
                    x0 = x0 + rng.normal(0, 0.01, size=N*3)
                    # Re-clip to ensure validity
                    x0_reshaped = x0.reshape(N, 3)
                    x0_reshaped = np.clip(x0_reshaped, -0.49, 0.49)
                    x0 = x0_reshaped.flatten()
                    
                result = minimize(
                    _neg_radius,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'fatol': 1e-12,
                        'xatol': 1e-12,
                        'adaptive': True
                    }
                )
                
                new_centers = result.x.reshape(N, 3)
                # Ensure centers are within bounds
                new_centers = np.clip(new_centers, -0.5, 0.5)
                
                new_r = _radius_for_centers(new_centers)
                
                if new_r > current_r:
                    current_r = new_r
                    current = new_centers.copy()
                else:
                    # If no improvement, break early
                    break
                    
            except Exception as e:
                # If optimization fails, break
                break
        
        if current_r > best_r:
            best_r = current_r
            best_centers = current.copy()
            
    # If we have time, do a final polish on the best center found
    if best_centers is not None and time.time() < deadline - 2.0:
        x0 = best_centers.flatten()
        try:
            result = minimize(
                _neg_radius,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'fatol': 1e-14,
                    'xatol': 1e-14,
                    'adaptive': True
                }
            )
            final_centers = result.x.reshape(N, 3)
            final_centers = np.clip(final_centers, -0.5, 0.5)
            final_r = _radius_for_centers(final_centers)
            
            if final_r > best_r:
                best_r = final_r
                best_centers = final_centers.copy()
        except Exception:
            pass

    # Fallback if something went wrong
    if best_centers is None:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
