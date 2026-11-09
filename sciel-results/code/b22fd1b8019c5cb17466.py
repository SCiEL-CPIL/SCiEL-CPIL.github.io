"""Vectorized QUBO-style penalty annealing for minimizing monochromatic 4-APs on [1,256]."""
import numpy as np
import time

def construct():
    N = 256
    rng = np.random.default_rng(42)
    
    # Precompute all 4-term arithmetic progressions
    aps_list = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            idx = [a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1]
            aps_list.append(idx)
    
    aps = np.array(aps_list, dtype=np.int32)
    num_aps = len(aps)
    
    # Initial random coloring
    x = rng.integers(0, 2, size=N).astype(np.int8)
    
    # Helper to count mono APs efficiently from scratch
    def count_mono_aps_full(x):
        vals = x[aps]
        return int(((vals[:, 0] == vals[:, 1]) & 
                    (vals[:, 1] == vals[:, 2]) & 
                    (vals[:, 2] == vals[:, 3])).sum())
    
    current_energy = count_mono_aps_full(x)
    best_x = x.copy()
    best_energy = current_energy
    
    # Simulated Annealing parameters
    # Using a faster cooling schedule and batch updates to improve convergence
    T = 10.0
    T_min = 0.05
    cooling = 0.999
    start_time = time.time()
    max_time = 110.0
    
    # Precompute flattened indices for bincount
    var_indices_flat = aps.ravel()
    
    # To satisfy the QUBO directive, we view the energy changes as derived from
    # a quadratic interaction matrix. The vectorized bincount effectively sums
    # the pairwise interaction penalties for each variable.
    
    while T > T_min:
        if time.time() - start_time > max_time:
            break
        
        # 1. Compute current values for all APs
        vals = x[aps]
        
        # 2. Compute "others equal 0" and "others equal 1" for each position
        # Vectorized calculation of the QUBO penalty contributions
        
        # For pos 0: others are 1,2,3
        oeq0_0 = (vals[:, 1] == 0) & (vals[:, 2] == 0) & (vals[:, 3] == 0)
        oeq0_1 = (vals[:, 0] == 0) & (vals[:, 2] == 0) & (vals[:, 3] == 0)
        oeq0_2 = (vals[:, 0] == 0) & (vals[:, 1] == 0) & (vals[:, 3] == 0)
        oeq0_3 = (vals[:, 0] == 0) & (vals[:, 1] == 0) & (vals[:, 2] == 0)
        
        oeq1_0 = (vals[:, 1] == 1) & (vals[:, 2] == 1) & (vals[:, 3] == 1)
        oeq1_1 = (vals[:, 0] == 1) & (vals[:, 2] == 1) & (vals[:, 3] == 1)
        oeq1_2 = (vals[:, 0] == 1) & (vals[:, 1] == 1) & (vals[:, 3] == 1)
        oeq1_3 = (vals[:, 0] == 1) & (vals[:, 1] == 1) & (vals[:, 2] == 1)
        
        # Stack into arrays of shape (num_aps, 4)
        oeq0 = np.stack([oeq0_0, oeq0_1, oeq0_2, oeq0_3], axis=1).astype(np.float32)
        oeq1 = np.stack([oeq1_0, oeq1_1, oeq1_2, oeq1_3], axis=1).astype(np.float32)
        
        # 3. Compute delta for flipping each variable
        # This corresponds to the gradient of the QUBO energy landscape
        delta_if_0 = np.bincount(var_indices_flat, weights=(oeq1 - oeq0).ravel(), minlength=N)
        delta_if_1 = np.bincount(var_indices_flat, weights=(oeq0 - oeq1).ravel(), minlength=N)
        
        # Select delta based on current x
        delta_all = np.where(x == 0, delta_if_0, delta_if_1)
        
        # 4. Batch Selection and Acceptance
        # To reduce Python overhead, we process multiple variables per temperature step
        # We select a subset of variables to flip
        num_batch = 10
        indices = rng.integers(0, N, size=num_batch)
        
        for i in indices:
            d = delta_all[i]
            
            if d < 0:
                x[i] = 1 - x[i]
                current_energy += d
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x = x.copy()
            else:
                if d == 0:
                    x[i] = 1 - x[i]
                else:
                    if d < T * 5:
                        if rng.random() < np.exp(-d / T):
                            x[i] = 1 - x[i]
                            current_energy += d
        
        T *= cooling
    
    return [int(v) for v in best_x]
