# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Returns base positions for 64 tubes.
    """
    N = 8
    start_time = time.time()
    time_limit = 600.0
    margin = 30.0
    
    # 1D Union Length Function
    # x: array of 8 base positions
    # c: scalar height
    # Intervals: [x[i] + c*i/N, x[i] + 1/N + c*i/N]
    def union_length_1d(x, c):
        # Create intervals
        # i ranges from 0 to 7
        i = np.arange(N)
        starts = x + c * i / N
        ends = starts + 1.0 / N
        
        # Sort by start
        idx = np.argsort(starts)
        starts = starts[idx]
        ends = ends[idx]
        
        # Compute union length
        length = 0.0
        current_end = starts[0]
        for i in range(1, N):
            if starts[i] <= current_end:
                current_end = max(current_end, ends[i])
            else:
                length += current_end - starts[i-1] # Wait, this logic is slightly off for the first segment
                # Let's do it properly:
                # length += current_end - previous_start? No.
                # Standard union length:
                # Sort intervals.
                # Iterate, keep track of current_max_end.
                # If next start > current_max_end, add current_max_end - current_start? No.
                # Let's restart the loop logic.
                pass
        
        # Correct logic for union length of sorted intervals
        length = 0.0
        curr_start = starts[0]
        curr_end = ends[0]
        for i in range(1, N):
            if starts[i] <= curr_end:
                curr_end = max(curr_end, ends[i])
            else:
                length += curr_end - curr_start
                curr_start = starts[i]
                curr_end = ends[i]
        length += curr_end - curr_start
        return length

    # Objective function for 1D part
    # A(x) = integral from 0 to 1 of union_length_1d(x, c) dc
    # We approximate the integral with a grid
    c_grid = np.linspace(0, 1, 100)
    
    def objective_1d(x):
        # x is flat array of 8 floats
        total_len = 0.0
        for c in c_grid:
            total_len += union_length_1d(x, c)
        return total_len / len(c_grid)

    # Initial guess: Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    # Normalize keich to be in a reasonable range, e.g., [0, 1]
    # The values are small. Let's shift and scale.
    # Actually, the relative positions matter.
    # Let's try to optimize starting from keich.
    x0 = keich.copy()
    
    # Optimization
    # Use Nelder-Mead as it doesn't require gradients and handles non-smoothness better
    res = opt.minimize(objective_1d, x0, method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    x_opt = res.x
    y_opt = x_opt.copy() # Symmetry
    
    # Check time
    if time.time() - start_time > time_limit - margin:
        # Fallback to initial guess if optimization took too long
        x_opt = x0
        y_opt = x0

    # Construct positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    # Sanity check: ensure finite and valid
    if not np.all(np.isfinite(positions)):
        # Fallback to simple grid if something went wrong
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
        
    return positions

# EVOLVE-BLOCK-END
