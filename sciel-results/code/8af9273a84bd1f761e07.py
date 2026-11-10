import numpy as np
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    # Baseline positions
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x_baseline = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x_baseline.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x = np.array(x_baseline, dtype=np.float64)
    
    # Precompute constants
    a = (np.arange(n) + 1) / n
    y = (np.arange(20000) + 0.5) / 20000
    w = (1 - y) / n
    # L = x_i + a_i * y_m
    # We need to compute area for a given x
    
    def compute_area(x):
        # L shape (20000, 64)
        L = x[None, :] + a[None, :] * y[:, None]
        L_sorted = np.sort(L, axis=1)
        diffs = np.diff(L_sorted, axis=1)
        # Union length at each y
        # w is shape (20000,)
        # minimum(diffs, w[:, None])
        union_lengths = w + np.sum(np.minimum(diffs, w[:, None]), axis=1)
        return np.mean(union_lengths)
    
    current_area = compute_area(x)
    start_time = time.time()
    deadline = start_time + 600 # Safety margin
    
    # Local search
    # We can try to improve by small perturbations
    # Since the function is piecewise linear, the optimum is likely at a vertex where some L_i's are equal.
    # Random search might get stuck in local minima.
    # Let's try a simple random restart or just many steps.
    
    step_size = 1e-3
    rng = np.random.default_rng(42)
    
    # To speed up, we can try to optimize a few variables at a time or use a smarter step size.
    # Let's do a simple hill climbing.
    
    for _ in range(50000): # Max iterations, but break on time
        if time.time() > deadline:
            break
        
        # Pick a random index
        i = rng.integers(0, n)
        
        # Try a small perturbation
        # Adaptive step size?
        delta = rng.normal(0, step_size)
        
        x_new = x.copy()
        x_new[i] += delta
        
        new_area = compute_area(x_new)
        
        if new_area < current_area:
            x = x_new
            current_area = new_area
            # Optionally shrink step size slightly
            # step_size *= 0.99
        else:
            # Revert
            pass
            
    return x
