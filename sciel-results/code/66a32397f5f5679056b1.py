# EVOLVE-BLOCK-START
import numpy as np
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    k = np.arange(64)
    i = k // N
    j = k % N
    
    # Initial positions: Keich construction (separable)
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    x_base = keich[i].copy()
    y_base = keich[j].copy()
    
    # Objective function: Monte Carlo Volume
    def compute_volume(x, y, n_samples=50000, seed=12345):
        X0 = np.min(x)
        X1 = np.max(x + (i + 1) / N)
        Y0 = np.min(y)
        Y1 = np.max(y + (j + 1) / N)
        
        if X1 - X0 < 1e-9 or Y1 - Y0 < 1e-9:
            return 1e9
            
        rng = np.random.default_rng(seed)
        a = rng.uniform(X0, X1, n_samples)
        b = rng.uniform(Y0, Y1, n_samples)
        c = rng.uniform(0, 1, n_samples)
        
        a_col = a[:, np.newaxis]
        b_col = b[:, np.newaxis]
        c_col = c[:, np.newaxis]
        
        x_row = x[np.newaxis, :]
        y_row = y[np.newaxis, :]
        i_row = i[np.newaxis, :]
        j_row = j[np.newaxis, :]
        
        x_min = x_row + c_col * i_row / N
        x_max = x_min + 1.0 / N
        y_min = y_row + c_col * j_row / N
        y_max = y_min + 1.0 / N
        
        in_x = (a_col >= x_min) & (a_col <= x_max)
        in_y = (b_col >= y_min) & (b_col <= y_max)
        
        in_tube = in_x & in_y
        in_union = np.any(in_tube, axis=1)
        
        fraction = np.mean(in_union)
        return fraction * (X1 - X0) * (Y1 - Y0)
    
    # Optimization Loop: Greedy Local Search
    start_time = time.time()
    deadline = start_time + 590.0
    
    x_best = x_base.copy()
    y_best = y_base.copy()
    v_best = compute_volume(x_best, y_best, n_samples=100000, seed=12345)
    
    step_size = 0.01
    
    while time.time() < deadline:
        improved = False
        for k_idx in range(64):
            if time.time() > deadline:
                break
                
            # Try moving x
            for dx in [-step_size, step_size]:
                x_trial = x_best.copy()
                x_trial[k_idx] += dx
                v_trial = compute_volume(x_trial, y_best, n_samples=5000, seed=12345 + k_idx)
                if v_trial < v_best:
                    x_best = x_trial
                    v_best = v_trial
                    improved = True
            
            # Try moving y
            for dy in [-step_size, step_size]:
                y_trial = y_best.copy()
                y_trial[k_idx] += dy
                v_trial = compute_volume(x_best, y_trial, n_samples=5000, seed=12345 + k_idx + 64)
                if v_trial < v_best:
                    y_best = y_trial
                    v_best = v_trial
                    improved = True
            
        if not improved:
            step_size *= 0.5
            if step_size < 1e-5:
                break
                
    positions = np.stack([x_best, y_best], axis=1)
    return positions
# EVOLVE-BLOCK-END
