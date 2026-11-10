# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def kakeya_needle_3d8() -> np.ndarray:
    """
    Optimizes the relative base positions of 64 sheared tubes to minimize the union volume.
    Fixes the first tube at (0,0) to remove translational invariance.
    Uses Differential Evolution with a reduced Monte Carlo sample size for speed.
    """
    N = 8
    num_tubes = N * N
    dim = num_tubes * 2 - 2  # Fix first x and y to 0
    
    # Pre-compute tube indices
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    # Reduced sample size for optimization speed.
    OPT_SAMPLES = 50000
    
    rng = np.random.default_rng(12345)
    u_a = rng.uniform(0, 1, OPT_SAMPLES)
    u_b = rng.uniform(0, 1, OPT_SAMPLES)
    u_c = rng.uniform(0, 1, OPT_SAMPLES)
    
    # Pre-compute constants for tube shifts
    # c_i = c * i_k / N
    # c_j = c * j_k / N
    # We will compute these inside the function as they depend on c (which is fixed sample)
    # but i_k and j_k are fixed.
    
    def calculate_volume(positions_flat):
        # positions_flat has 126 elements.
        # Reconstruct full positions (64, 2)
        positions = np.zeros((num_tubes, 2))
        # First tube is (0,0)
        # The rest are mapped from positions_flat
        # positions_flat[0] -> x_1, positions_flat[1] -> y_1, ...
        # Wait, the order is row-major: (0,0), (0,1), ..., (0,7), (1,0)...
        # We fix (0,0) to (0,0).
        # So positions[0] = [0,0]
        # positions[1:] are filled from positions_flat
        
        positions[1:, 0] = positions_flat[0::2]
        positions[1:, 1] = positions_flat[1::2]
        
        x = positions[:, 0]
        y = positions[:, 1]
        
        # Calculate bounding box
        x_min = np.min(x)
        x_max = np.max(x + (i_flat + 1) / N)
        y_min = np.min(y)
        y_max = np.max(y + (j_flat + 1) / N)
        
        box_w = x_max - x_min
        box_h = y_max - y_min
        
        if box_w <= 1e-9 or box_h <= 1e-9 or not np.isfinite(box_w) or not np.isfinite(box_h):
            return 1e10
        
        # Map uniform samples to the current box
        a = x_min + u_a * box_w
        b = y_min + u_b * box_h
        c = u_c
        
        # Efficient check for inclusion
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
        
        any_inside = np.any(inside_x & inside_y, axis=1)
        
        fraction = np.mean(any_inside)
        volume = fraction * box_w * box_h
        
        if not np.isfinite(volume) or volume <= 0:
            return 1e10
            
        return volume

    # Generate baseline Keich construction for initialization
    pos_int = int(np.log2(N))
    j_terms = np.arange(1, pos_int + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_int - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_int)
        
    base_x = keich[i_flat]
    base_y = keich[j_flat]
    base_pos_full = np.stack([base_x, base_y], axis=1)
    
    # Relative positions for baseline (subtract first element)
    base_rel = base_pos_full - base_pos_full[0]
    # Flatten and remove first two elements
    baseline_flat = base_rel.flatten()
    baseline_flat = np.delete(baseline_flat, [0, 1]) # Remove x0, y0
    
    # Bounds
    # Relative shifts shouldn't be too large. 1.0 is a good bound.
    bounds = [(-1.0, 1.0)] * dim
    
    # Initial population
    # We can create a custom init with the baseline
    # But DE's 'latinhypercube' is good.
    # Let's use 'sobol' if possible, otherwise 'latinhypercube'.
    
    result = differential_evolution(
        calculate_volume, 
        bounds, 
        seed=42,
        maxiter=100,
        popsize=20,
        tol=0.01,
        mutation=(0.5, 1.0),
        recombination=0.7,
        init='latinhypercube',
        polish=False
    )
    
    best_pos_flat = result.x
    best_vol = result.fun
    
    if not np.isfinite(best_vol) or best_vol > 1e9:
        best_pos_flat = baseline_flat
        
    # Reconstruct full positions
    best_positions = np.zeros((num_tubes, 2))
    best_positions[1:, 0] = best_pos_flat[0::2]
    best_positions[1:, 1] = best_pos_flat[1::2]
    
    return best_positions

# EVOLVE-BLOCK-END
