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
    
    # 1D Union Length Function
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
        
        # Compute union length using sweep line
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

    # Exact integration of union length over c in [0,1]
    # The union length is piecewise linear in c, with breakpoints where interval endpoints coincide
    # For N=8, we can compute this exactly by finding all breakpoints and integrating piecewise
    
    def objective_1d_exact(x):
        """
        Compute the exact integral of union_length_1d(x, c) dc from 0 to 1.
        """
        # Get all interval endpoints as functions of c
        # Interval i: [x[i] + c*i/N, x[i] + 1/N + c*i/N]
        # Endpoints are linear in c: a + b*c where a = x[i] or x[i]+1/N, b = i/N
        
        # Collect all endpoint lines (a, b) where endpoint = a + b*c
        lines = []
        for i in range(N):
            lines.append((x[i], i/N))           # start of interval i
            lines.append((x[i] + 1.0/N, i/N))   # end of interval i
        
        # Find all c values where two endpoints are equal
        # a1 + b1*c = a2 + b2*c => c = (a2-a1)/(b1-b2) if b1 != b2
        breakpoints = set([0.0, 1.0])
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                a1, b1 = lines[i]
                a2, b2 = lines[j]
                if abs(b1 - b2) > 1e-12:
                    c_val = (a2 - a1) / (b1 - b2)
                    if 0.0 <= c_val <= 1.0:
                        breakpoints.add(c_val)
        
        # Sort breakpoints
        breakpoints = sorted(breakpoints)
        
        # For each interval between consecutive breakpoints, the ordering of endpoints is constant
        # So the union length is linear in c in each such interval
        # We can evaluate at the midpoint and use trapezoidal rule (exact for linear functions)
        
        total = 0.0
        for k in range(len(breakpoints) - 1):
            c_left = breakpoints[k]
            c_right = breakpoints[k+1]
            c_mid = (c_left + c_right) / 2.0
            
            len_left = union_length_1d(x, c_left)
            len_right = union_length_1d(x, c_right)
            len_mid = union_length_1d(x, c_mid)
            
            # Simpson's rule or trapezoidal
            # Since union length is piecewise linear, trapezoidal is exact
            total += (len_left + len_right) / 2.0 * (c_right - c_left)
        
        return total

    # Optimization
    # Use Nelder-Mead with exact objective
    
    # Initial guess: Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    x0 = keich.copy()
    
    # First optimization pass
    res = opt.minimize(objective_1d_exact, x0, method='Nelder-Mead', 
                       options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
    
    x_opt = res.x
    
    # Check time
    if time.time() - start_time > time_limit - margin:
        # Fallback to initial guess if optimization took too long
        x_opt = x0
        y_opt = x0
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        i_flat = i_grid.flatten()
        j_flat = j_grid.flatten()
        base_x = x_opt[i_flat]
        base_y = y_opt[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
        return positions
    
    # Second optimization pass with tighter tolerance
    res2 = opt.minimize(objective_1d_exact, x_opt, method='Nelder-Mead',
                        options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
    
    x_opt = res2.x
    
    # Check time again
    if time.time() - start_time > time_limit - margin:
        y_opt = x_opt.copy()
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        i_flat = i_grid.flatten()
        j_flat = j_grid.flatten()
        base_x = x_opt[i_flat]
        base_y = y_opt[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
        return positions
    
    # Try asymmetric x and y
    # Optimize y separately
    def objective_1d_y(y):
        return objective_1d_exact(y)
    
    y0 = x_opt.copy()
    res_y = opt.minimize(objective_1d_y, y0, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
    
    y_opt = res_y.x
    
    # Check time
    if time.time() - start_time > time_limit - margin:
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        i_flat = i_grid.flatten()
        j_flat = j_grid.flatten()
        base_x = x_opt[i_flat]
        base_y = y_opt[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
        return positions
    
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
