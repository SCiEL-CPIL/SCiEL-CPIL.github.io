import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    # 1. Generate Pool
    pool = []
    
    # Add weight-1 vectors (norm 2)
    for i in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[i] = sign * 2
            pool.append(v)
            
    # Add weight-4 vectors (norm 2)
    indices = range(D)
    for comb in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            pool.append(v)
            
    pool = np.array(pool, dtype=np.int64)
    n_pool = len(pool)
    
    # 2. Precompute Compatibility
    # dots[i, j] = pool[i] . pool[j]
    # Compatible if dots[i, j] <= 2
    dots = pool @ pool.T
    compatible = dots <= 2
    
    # 3. Greedy Initialization
    x = np.zeros(n_pool, dtype=bool)
    indices_greedy = np.arange(n_pool)
    np.random.shuffle(indices_greedy)
    
    for i in indices_greedy:
        if time.time() > deadline - 10:
            break
        if np.all(compatible[i, x]):
            x[i] = True
            
    current_score = np.sum(x)
    best_x = x.copy()
    best_score = current_score
    
    # 4. Simulated Annealing
    T_start = 5.0
    T_end = 0.1
    cooling_rate = 0.999
    T = T_start
    
    rng = np.random.default_rng(42)
    
    while T > T_end and time.time() < deadline - 2:
        move_type = rng.random()
        
        if move_type < 0.8:
            # Try Swap
            selected_indices = np.where(x)[0]
            unselected_indices = np.where(~x)[0]
            
            if len(selected_indices) > 0 and len(unselected_indices) > 0:
                i = selected_indices[rng.integers(len(selected_indices))]
                j = unselected_indices[rng.integers(len(unselected_indices))]
                
                # Check if j is compatible with all selected except i
                # incomp = number of selected k such that not compatible[j, k]
                incomp = np.sum(~compatible[j, x])
                
                if incomp == 0:
                    # j compatible with all. Swap is valid.
                    x[i] = False
                    x[j] = True
                elif incomp == 1:
                    # Check if the only incompatible one is i
                    if not compatible[j, i]:
                        x[i] = False
                        x[j] = True
        else:
            # Try Remove
            selected_indices = np.where(x)[0]
            if len(selected_indices) > 0:
                i = selected_indices[rng.integers(len(selected_indices))]
                # Accept removal with prob exp(-1/T)
                if rng.random() < np.exp(-1.0 / T):
                    x[i] = False
                    # Local Repair: Try to add any unselected vector that is now compatible
                    # To save time, we check a random subset or just loop.
                    # Looping through all unselected is O(n_pool * n_selected) ~ 5k * 100 = 500k ops.
                    # This is fast enough for one step.
                    unselected_indices = np.where(~x)[0]
                    for j in unselected_indices:
                        if np.all(compatible[j, x]):
                            x[j] = True
                            break # Add one, then re-evaluate score
                    
        # Update Best
        current_score = np.sum(x)
        if current_score > best_score:
            best_score = current_score
            best_x = x.copy()
            
        # Cooling
        T *= cooling_rate
        
    # Return the selected vectors
    selected_vectors = pool[best_x]
    return selected_vectors

# EVOLVE-BLOCK-END
