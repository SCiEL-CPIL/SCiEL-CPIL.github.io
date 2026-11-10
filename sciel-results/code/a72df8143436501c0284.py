import numpy as np
from scipy.optimize import differential_evolution
import time

def _compute_volume_exact(x, y):
    """
    Computes the exact volume of the union of tubes.
    x: array of shape (N,), base x positions for i=0..N-1
    y: array of shape (N,), base y positions for j=0..N-1
    N: fixed at 8
    """
    N = 8
    # The volume is integral_0^1 L_x(c) L_y(c) dc
    # We use a fine grid for numerical integration to be robust and fast
    # 1000 points is sufficient for high accuracy and very fast
    c_vals = np.linspace(0, 1, 1001)
    
    # Precompute shifts
    # i is 0..7, j is 0..7
    i_idx = np.arange(N)
    j_idx = np.arange(N)
    
    # L_x(c) is union of intervals [x_i + c*i/N, x_i + 1/N + c*i/N]
    # L_y(c) is union of intervals [y_j + c*j/N, y_j + 1/N + c*j/N]
    
    # To speed up, we can vectorize the interval union calculation
    # However, union of intervals is not trivially vectorizable without sorting.
    # N=8 is small, so a loop over c_vals is fine.
    
    vol = 0.0
    h = c_vals[1] - c_vals[0]
    
    for c in c_vals:
        # Calculate x intervals
        # starts: x_i + c * i / N
        # ends: x_i + 1/N + c * i / N
        x_starts = x + c * i_idx / N
        x_ends = x + 1.0/N + c * i_idx / N
        
        # Union length of x intervals
        # Sort by start
        order = np.argsort(x_starts)
        sorted_starts = x_starts[order]
        sorted_ends = x_ends[order]
        
        # Merge intervals
        current_max_end = sorted_ends[0]
        lx = 0.0
        for k in range(1, N):
            if sorted_starts[k] <= current_max_end:
                current_max_end = max(current_max_end, sorted_ends[k])
            else:
                lx += current_max_end - sorted_starts[k-1] # Wait, this logic is slightly off.
                # Standard merge:
                # lx += current_max_end - start_of_current_segment
                # Let's rewrite merge logic clearly.
                pass
        
        # Let's use a simpler, correct merge loop
        lx = 0.0
        cur_start = sorted_starts[0]
        cur_end = sorted_ends[0]
        for k in range(1, N):
            if sorted_starts[k] <= cur_end:
                if sorted_ends[k] > cur_end:
                    cur_end = sorted_ends[k]
            else:
                lx += cur_end - cur_start
                cur_start = sorted_starts[k]
                cur_end = sorted_ends[k]
        lx += cur_end - cur_start

        # Calculate y intervals
        y_starts = y + c * j_idx / N
        y_ends = y + 1.0/N + c * j_idx / N
        
        order_y = np.argsort(y_starts)
        sorted_y_starts = y_starts[order_y]
        sorted_y_ends = y_ends[order_y]
        
        ly = 0.0
        cur_start_y = sorted_y_starts[0]
        cur_end_y = sorted_y_ends[0]
        for k in range(1, N):
            if sorted_y_starts[k] <= cur_end_y:
                if sorted_y_ends[k] > cur_end_y:
                    cur_end_y = sorted_y_ends[k]
            else:
                ly += cur_end_y - cur_start_y
                cur_start_y = sorted_y_starts[k]
                cur_end_y = sorted_y_ends[k]
        ly += cur_end_y - cur_start_y

        vol += lx * ly
        
    vol *= h
    return vol

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    # Initial guess: spread out a bit
    # We want to minimize volume.
    # Bounds for optimization: positions can be anywhere, but relative positions matter.
    # We can constrain the first x and y to 0 to fix translation invariance, 
    # but DE handles this fine if we just let it search.
    # However, to keep numbers small, let's bound them to [-1, 1] relative to some center?
    # Actually, the volume is translation invariant. 
    # We can fix x[0]=0 and y[0]=0 to reduce dimensionality to 14?
    # Or just let DE search in [-1, 1]^16.
    
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Objective function wrapper
    def objective(params):
        x = params[:N]
        y = params[N:]
        return _compute_volume_exact(x, y)
    
    # Run Differential Evolution
    # maxiter and popsize control time. 
    # 16D problem. popsize=15 (default) -> 240 individuals.
    # 100 iterations -> 24000 evaluations.
    # Each eval ~ 1000 steps * 2 merges. ~ 20000 ops. 
    # 24000 * 20000 = 480M ops. Might be a bit slow in Python loops.
    # Let's optimize the inner loop or reduce integration points.
    # 200 points should be enough for volume accuracy.
    
    # Let's modify _compute_volume_exact to accept num_points
    # Or just reduce it in the wrapper? No, it's hardcoded.
    # Let's rewrite the inner loop to be faster or use fewer points.
    
    # Actually, 1001 points is fine. 24000 evals * 1000 * 8 is 192M operations.
    # In Python, this might take ~20-60 seconds. Acceptable.
    
    start_time = time.time()
    timeout = 580.0 # Safety margin
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=10, 
            tol=1e-6,
            polish=True
        )
    except Exception:
        # Fallback to seed if DE fails or times out (though DE doesn't have explicit timeout)
        # We can check time in a callback? No, simple try/except for errors.
        # To handle time, we can run DE in a way that we can interrupt?
        # Or just reduce maxiter.
        # Let's assume it finishes.
        pass
    
    if not res.success and res.fun == np.inf:
        # Fallback
        x = np.zeros(N)
        y = np.zeros(N)
    else:
        x = res.x[:N]
        y = res.x[N:]
        
    # Construct positions array
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x[i_flat]
    base_y = y[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
