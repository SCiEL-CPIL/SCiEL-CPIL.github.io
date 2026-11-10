# EVOLVE-BLOCK-START
import numpy as np
import time

def _keich_positions_1d(N):
    """Keich binary construction for N = 2^k base positions."""
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich

def _mc_volume_fast(base_x, base_y, N=8, n_samples=500000, seed=12345):
    """
    Fast Monte Carlo volume estimate.
    base_x: shape (N,) - base x for each i
    base_y: shape (N,) - base y for each j
    Returns the estimated union volume.
    """
    rng = np.random.default_rng(seed)
    
    # Compute bounding box
    # x_k = base_x[i], tube x-interval at c: [x_k + c*i/N, x_k + c*i/N + 1/N]
    # X0 = min_k x_k = min(base_x)
    # X1 = max_k (x_k + (i+1)/N)  ... wait, the contract says X1 = max_k (x_k + (i+1)/8)
    # Let's re-read: "X1 = max_k (x_k + (i+1)/8)"
    # But the tube is [x_k + c*i/8, x_k + c*i/8 + 1/8].
    # At c=1, the interval is [x_k + i/8, x_k + i/8 + 1/8] = [x_k + (i+1)/8 - 1/8, x_k + (i+1)/8].
    # So the max x is indeed max(x_k + (i+1)/8).
    
    X0 = np.min(base_x)
    X1 = np.max(base_x + (np.arange(N) + 1) / N)
    
    Y0 = np.min(base_y)
    Y1 = np.max(base_y + (np.arange(N) + 1) / N)
    
    area = (X1 - X0) * (Y1 - Y0)
    if area <= 1e-9:
        return 1e9 # Invalid/zero area
    
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    # Check if point (a, b, c) is inside at least one tube.
    # Tube (i, j) contains (a, b, c) if:
    # x_k + c*i/N <= a <= x_k + c*i/N + 1/N
    # y_k + c*j/N <= b <= y_k + c*j/N + 1/N
    
    # We can vectorize this.
    # For each sample, we need to check if there exists i, j such that the conditions hold.
    # Since N=8, we can just loop over i and j, or use broadcasting.
    # Broadcasting: 
    # a: (M,)
    # c: (M,)
    # base_x: (N,)
    # We want to check for each sample m, if any i satisfies:
    # base_x[i] + c[m]*i/N <= a[m] <= base_x[i] + c[m]*i/N + 1/N
    
    # Let's create arrays:
    # a_expanded: (M, 1)
    # c_expanded: (M, 1)
    # base_x_expanded: (1, N)
    # i_expanded: (1, N)
    
    a_e = a[:, np.newaxis]
    b_e = b[:, np.newaxis]
    c_e = c[:, np.newaxis]
    
    bx_e = base_x[np.newaxis, :]
    by_e = base_y[np.newaxis, :]
    i_e = np.arange(N)[np.newaxis, :]
    j_e = np.arange(N)[np.newaxis, :]
    
    # x_start = bx_e + c_e * i_e / N  (M, N)
    # x_end = x_start + 1/N
    # y_start = by_e + c_e * j_e / N
    # y_end = y_start + 1/N
    
    x_start = bx_e + c_e * i_e / N
    y_start = by_e + c_e * j_e / N
    
    # Condition: x_start <= a <= x_start + 1/N  AND  y_start <= b <= y_start + 1/N
    # Note: The tube is a product of x-interval (depends on i) and y-interval (depends on j).
    # So (a,b,c) is in tube (i,j) iff (a is in x-interval of i at c) AND (b is in y-interval of j at c).
    # This means we can check x and y independently for each sample.
    
    # x_in[i, m] = (x_start[i, m] <= a[m]) & (a[m] <= x_start[i, m] + 1/N)
    # x_in shape (M, N)
    x_in = (x_start <= a_e) & (a_e <= x_start + 1.0/N)
    y_in = (y_start <= b_e) & (b_e <= y_start + 1.0/N)
    
    # A sample is covered if there exists i such that x_in[m, i] is True AND exists j such that y_in[m, j] is True.
    # Because the tube is a Cartesian product of the x-interval for i and y-interval for j.
    # So if a is in some x-interval and b is in some y-interval, the point is in the union.
    
    x_covered = np.any(x_in, axis=1)
    y_covered = np.any(y_in, axis=1)
    
    covered = x_covered & y_covered
    
    fraction = np.mean(covered)
    V = fraction * area
    
    if not np.isfinite(V) or V <= 0:
        return 1e9
        
    return V

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    t_start = time.time()
    deadline = t_start + 600.0
    
    # Initial positions from Keich
    keich = _keich_positions_1d(N)
    base_x = keich.copy()
    base_y = keich.copy()
    
    # Calculate initial volume
    current_V = _mc_volume_fast(base_x, base_y, N, 500000, 12345)
    best_V = current_V
    best_x = base_x.copy()
    best_y = base_y.copy()
    
    # Local search optimization
    # We will perform a simple coordinate descent or random hill climbing.
    # Given the time limit, we should be careful.
    # Let's try a few random perturbations and keep the best.
    # Or better, use a small number of steps of a gradient-free method.
    # Since the objective is non-smooth (MC estimate), stochastic gradient is noisy.
    # Let's try a few iterations of random search.
    
    rng_opt = np.random.default_rng(42)
    
    # We have 16 variables (8 x, 8 y).
    # Let's try to optimize them.
    
    # Strategy: Random restarts with local refinement?
    # Or just a long random walk?
    
    # Let's try a simple hill climbing with small steps.
    step_size = 0.1
    iterations = 0
    max_iterations = 5000 # Just a cap, time is the real limit
    
    while time.time() < deadline - 10.0:
        iterations += 1
        
        # Pick a random variable to perturb
        idx = rng_opt.integers(0, 16)
        if idx < 8:
            # Perturb base_x[idx]
            delta = rng_opt.uniform(-step_size, step_size)
            new_x = base_x.copy()
            new_x[idx] += delta
            # Check bounds? Not strictly necessary, but keep them reasonable.
            # The bounding box will adjust.
            new_V = _mc_volume_fast(new_x, base_y, N, 50000, 12345) # Fewer samples for speed during search
            if new_V < current_V:
                base_x = new_x
                current_V = new_V
                if current_V < best_V:
                    best_V = current_V
                    best_x = base_x.copy()
                    best_y = base_y.copy()
            # Decay step size slightly
            step_size *= 0.999
        else:
            # Perturb base_y[idx-8]
            idx_y = idx - 8
            delta = rng_opt.uniform(-step_size, step_size)
            new_y = base_y.copy()
            new_y[idx_y] += delta
            new_V = _mc_volume_fast(base_x, new_y, N, 50000, 12345)
            if new_V < current_V:
                base_y = new_y
                current_V = new_V
                if current_V < best_V:
                    best_V = current_V
                    best_x = base_x.copy()
                    best_y = base_y.copy()
            step_size *= 0.999
            
    # Final high-precision check on the best found
    # Re-evaluate with full samples to be sure
    final_V = _mc_volume_fast(best_x, best_y, N, 500000, 12345)
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x_final = best_x[i_flat]
    base_y_final = best_y[j_flat]
    positions = np.stack([base_x_final, base_y_final], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
