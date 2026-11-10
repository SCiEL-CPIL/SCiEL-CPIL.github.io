# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def kakeya_needle_3d8() -> np.ndarray:
    """
    Optimizes the base positions of 64 sheared tubes to minimize the union volume.
    Uses Differential Evolution on a deterministic Monte Carlo volume estimator.
    """
    N = 8
    num_tubes = N * N
    dim = num_tubes * 2
    
    # Pre-compute the fixed random samples for the Monte Carlo method.
    rng = np.random.default_rng(12345)
    u_a = rng.uniform(0, 1, 500000)
    u_b = rng.uniform(0, 1, 500000)
    u_c = rng.uniform(0, 1, 500000)
    
    # Pre-compute tube indices
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    # Helper to calculate volume for a given set of positions
    def calculate_volume(positions_flat):
        # Reshape to (64, 2)
        positions = positions_flat.reshape(num_tubes, 2)
        x = positions[:, 0]
        y = positions[:, 1]
        
        # Calculate bounding box
        x_min = np.min(x)
        x_max = np.max(x + (i_flat + 1) / N)
        
        y_min = np.min(y)
        y_max = np.max(y + (j_flat + 1) / N)
        
        box_w = x_max - x_min
        box_h = y_max - y_min
        
        # Invalid if box is too small or non-finite
        if box_w <= 1e-9 or box_h <= 1e-9 or not np.isfinite(box_w) or not np.isfinite(box_h):
            return 1e10
        
        # Map uniform samples to the current box
        a = x_min + u_a * box_w
        b = y_min + u_b * box_h
        c = u_c
        
        # Check if points are inside any tube
        c_i = c[:, None] * (i_flat[None, :] / N)
        c_j = c[:, None] * (j_flat[None, :] / N)
        
        x_start = x[None, :] + c_i
        x_end = x_start + 1.0 / N
        y_start = y[None, :] + c_j
        y_end = y_start + 1.0 / N
        
        a_exp = a[:, None]
        b_exp = b[:, None]
        
        inside_x = (a_exp >= x_start) & (a_exp <= x_end)
        inside_y = (b_exp >= y_start) & (b_exp <= y_end)
        
        inside_all = inside_x & inside_y
        
        any_inside = np.any(inside_all, axis=1)
        
        fraction = np.mean(any_inside)
        volume = fraction * box_w * box_h
        
        if not np.isfinite(volume) or volume <= 0:
            return 1e10
            
        return volume

    # Generate baseline positions using Keich construction
    pos_int = int(np.log2(N))
    j_terms = np.arange(1, pos_int + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_int - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_int)
        
    base_x = keich[i_flat]
    base_y = keich[j_flat]
    baseline_pos = np.stack([base_x, base_y], axis=1).flatten()
    
    # Use Differential Evolution with improved parameters
    bounds = [(-1.5, 1.5)] * dim
    
    # Create initial population centered around baseline
    popsize = 50
    init_pop = np.random.uniform(-0.5, 0.5, size=(popsize, dim))
    init_pop[0] = baseline_pos
    # Add a few more good seeds
    for k in range(1, 5):
        init_pop[k] = baseline_pos + np.random.normal(0, 0.2, size=dim)
    
    # Run Differential Evolution with more iterations and better settings
    result = differential_evolution(
        calculate_volume, 
        bounds, 
        seed=42,
        maxiter=100,
        popsize=popsize,
        tol=0.005,
        mutation=(0.3, 1.0),
        recombination=0.8,
        init=init_pop,
        polish=False,
        update_interval=0
    )
    
    best_pos = result.x
    best_vol = result.fun
    
    # If DE failed or returned invalid, fallback to baseline
    if not np.isfinite(best_vol) or best_vol > 1e9:
        best_pos = baseline_pos
        
    positions = best_pos.reshape(num_tubes, 2)
    
    return positions

# EVOLVE-BLOCK-END
