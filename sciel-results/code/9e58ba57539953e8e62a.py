import numpy as np
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    a = np.arange(1, n + 1) / n  # shape (64,)
    
    # Exact evaluation grid
    num_h = 20000
    ys = (np.arange(num_h) + 0.5) / num_h          # shape (20000,)
    ws = (1.0 - ys) / n                             # shape (20000,)

    def union_area_exact(x):
        """
        Exact union-area over the 20000 y-grid.
        x: shape (64,)
        Returns scalar area.
        """
        # Ls[y, i] = x[i] + a[i] * ys[y]  ->  shape (20000, 64)
        Ls = x[None, :] + a[None, :] * ys[:, None]  
        # Sort along axis=1
        sorted_Ls = np.sort(Ls, axis=1)             
        diffs = np.diff(sorted_Ls, axis=1)          
        # extensions[y, k] = min(ws[y], diffs[y, k])
        extensions = np.minimum(ws[:, None], diffs) 
        union_lens = ws + extensions.sum(axis=1)    
        return union_lens.mean()

    # Keich seed
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    seed_positions = np.zeros(n, dtype=np.float64)
    for i in range(n):
        bits = np.array(list(str(bin(i))[2:]), dtype=np.int8)
        bits = np.pad(bits, (k - len(bits), 0), constant_values=0)
        seed_positions[i] = np.sum((1.0 - j) * bits * (2.0 ** -j) / k)

    start_time = time.time()
    max_time = 620.0  # safety margin

    x = seed_positions.copy()
    best_x = x.copy()
    current_area = union_area_exact(x)
    best_area = current_area
    
    np.random.seed(42)
    
    # Simulated Annealing parameters
    T_start = 0.05
    T_end = 1e-6
    # We estimate the number of iterations we can do.
    # Each eval is ~5-10ms. Let's aim for ~3000 iterations to be safe, but loop until time runs out.
    
    # Initial step size
    step = 0.05
    
    # We will use a geometric cooling schedule.
    # Let's determine the number of iterations based on a time budget for the loop body.
    # We'll just run as many as we can.
    
    # To make SA effective, we need a good temperature schedule.
    # Let's assume we can do N_iter iterations.
    # T(t) = T_start * (T_end/T_start)^(t/N_iter)
    
    # Since we don't know N_iter exactly, we'll use a fixed number of "phases" or just a long run.
    # Let's target ~5000 iterations.
    target_iters = 5000
    alpha = (T_end / T_start) ** (1.0 / target_iters)
    
    T = T_start
    
    for i_iter in range(100000): # Large cap, will break on time
        if time.time() - start_time > max_time - 2.0:
            break
            
        # Generate a neighbor
        # Perturb a random subset of coordinates or all?
        # Perturbing all with small noise is often better for smooth landscapes.
        # Perturbing a single coordinate is better for rugged ones.
        # Let's try perturbing all with Gaussian noise.
        x_new = x + np.random.normal(0, step, n)
        
        new_area = union_area_exact(x_new)
        
        delta = new_area - current_area
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            x = x_new
            current_area = new_area
            if current_area < best_area:
                best_area = current_area
                best_x = x.copy()
        
        # Update temperature
        T *= alpha
        
        # Adapt step size?
        # If we accept too many moves, increase step. If too few, decrease.
        # Simple heuristic: keep step constant or decay slightly.
        # Let's decay step size slightly to allow fine-tuning.
        step *= 0.9999

    return np.array(best_x, dtype=np.float64)
