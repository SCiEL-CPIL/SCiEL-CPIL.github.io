import numpy as np
import time
import sys
from scipy.optimize import differential_evolution

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = 600.0 - 5.0  # Safety margin
    
    N = 8
    # We want to find x_0..x_7 to minimize the 1D union length integral.
    # Then use the same for y.
    
    # 1D Union Length calculation for a given x vector and c
    def union_length(x, c):
        # x: shape (8,)
        # c: scalar
        # Returns length of union of intervals
        starts = x + c * np.arange(N) / N
        ends = starts + 1.0 / N
        # Sort by start
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        
        # Merge
        total_len = 0.0
        current_end = -np.inf
        for s, e in zip(s_sorted, e_sorted):
            if s > current_end:
                total_len += (e - s)
                current_end = e
            else:
                if e > current_end:
                    total_len += (e - current_end)
                    current_end = e
        return total_len

    # Cost function for optimizer
    def cost(x):
        # x: shape (8,)
        # Sample c in [0, 1]
        # Use fixed random samples for consistency during optimization
        # Or deterministic quadrature? 
        # Let's use a fixed set of c values
        # We need to define this outside or use a global
        pass

    # Define fixed c samples
    c_samples = np.linspace(0.0, 1.0, 200)
    
    def cost_func(x):
        # x is flattened array of 8 floats
        total = 0.0
        for c in c_samples:
            total += union_length(x, c)
        return total / len(c_samples)

    # Optimization
    # Bounds: 
    # The intervals move by at most 1.0 (i/8 * 1). 
    # Base x should be in a range that allows overlap.
    # If x_i are too far apart, union length is 1.0 (sum of 1/8).
    # We want them close.
    # Range [-2, 2] should be sufficient.
    bounds = [(-2.0, 2.0) for _ in range(N)]
    
    # Run differential evolution
    # maxiter and popsize control time
    # We check time in a wrapper? DE doesn't support callbacks easily for stopping.
    # We can run a few iterations and check time.
    # Or just run with conservative params.
    
    # Let's try to run DE.
    # If it takes too long, we return the best found so far?
    # DE in scipy doesn't have a built-in time limit, but we can monitor.
    # Alternatively, use a smaller population and iterations.
    
    try:
        result = differential_evolution(
            cost_func, 
            bounds, 
            maxiter=50, 
            popsize=20, 
            tol=1e-6, 
            seed=42,
            init='latinhypercube'
        )
    except Exception as e:
        # Fallback to seed construction if optimization fails
        pos = int(np.log2(N))
        j_terms = np.arange(1, pos + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        i_flat = i_grid.flatten()
        j_flat = j_grid.flatten()
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        return np.stack([base_x, base_y], axis=1)

    if result.success:
        x_opt = result.x
    else:
        # Use the best x found even if not fully converged
        x_opt = result.x

    # Shift x_opt so that min(x_opt) is 0? 
    # The problem is translation invariant for the volume fraction, 
    # but the bounding box size depends on the spread.
    # Shifting doesn't change spread.
    # Let's keep x_opt as is.
    
    # Construct positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = x_opt[j_flat] # Assume symmetric optimal solution
    
    positions = np.stack([base_x, base_y], axis=1)
    
    # Validate
    if not np.all(np.isfinite(positions)):
        # Fallback
        pos = int(np.log2(N))
        j_terms = np.arange(1, pos + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
        
    return positions
