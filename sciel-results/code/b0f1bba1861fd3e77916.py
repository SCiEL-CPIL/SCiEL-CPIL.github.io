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
    margin = 60.0
    
    # 1D Union Length Function - cleaned up
    # x: array of 8 base positions
    # c: scalar height
    # Intervals: [x[i] + c*i/N, x[i] + 1/N + c*i/N]
    def union_length_1d(x, c):
        # Create intervals
        i = np.arange(N)
        starts = x + c * i / N
        ends = starts + 1.0 / N
        
        # Sort by start
        idx = np.argsort(starts)
        starts = starts[idx]
        ends = ends[idx]
        
        # Compute union length
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
    # We approximate the integral with a finer grid
    c_grid = np.linspace(0, 1, 200)
    
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
    
    # Normalize keich to be in a reasonable range
    x0 = keich.copy()
    
    # Multiple restarts with different initial points
    best_x = None
    best_obj = np.inf
    
    # Generate several initial guesses
    rng = np.random.default_rng(42)
    initial_points = [x0.copy()]
    
    # Add some random perturbations
    for _ in range(5):
        perturbed = x0 + rng.uniform(-0.1, 0.1, size=N)
        initial_points.append(perturbed)
    
    # Add a few more diverse starting points
    initial_points.append(np.linspace(0, 1, N))
    initial_points.append(np.random.default_rng(123).uniform(0, 1, size=N))
    
    for x_init in initial_points:
        if time.time() - start_time > time_limit - margin:
            break
        
        try:
            res = opt.minimize(objective_1d, x_init, method='Nelder-Mead', 
                              options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-7})
            
            obj_val = objective_1d(res.x)
            if obj_val < best_obj:
                best_obj = obj_val
                best_x = res.x.copy()
        except Exception:
            continue
    
    if best_x is None:
        best_x = x0
    
    # Final polish pass with tighter tolerance
    if time.time() - start_time < time_limit - margin:
        try:
            res_polish = opt.minimize(objective_1d, best_x, method='Nelder-Mead',
                                     options={'maxiter': 10000, 'xatol': 1e-9, 'fatol': 1e-9})
            if objective_1d(res_polish.x) < best_obj:
                best_x = res_polish.x
        except Exception:
            pass
    
    x_opt = best_x
    y_opt = x_opt.copy()  # Symmetry
    
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
