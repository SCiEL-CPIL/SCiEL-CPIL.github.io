import time
import numpy as np
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Generate pool of integer vectors with norm squared = 8
    # Type A: Two coordinates are +/-2, rest 0
    type_a = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    type_a.append(v)
    
    # Type B: Four coordinates are +/-1, rest 0
    type_b = []
    # Generate all combinations of 4 indices
    for combo in itertools.combinations(range(D), 4):
        i1, i2, i3, i4 = combo
        # 16 sign combinations
        for s1 in (1, -1):
            for s2 in (1, -1):
                for s3 in (1, -1):
                    for s4 in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i1] = s1
                        v[i2] = s2
                        v[i3] = s3
                        v[i4] = s4
                        type_b.append(v)
                        
    pool = np.array(type_a + type_b, dtype=np.int64)
    
    # Greedy selection
    # To improve results, try a few random permutations of the pool
    # or a specific heuristic (e.g., sort by something?)
    # Random restarts are good for independent set approximations.
    
    best_set = []
    n_restarts = 5
    np.random.seed(42)
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Shuffle pool
        idx = np.random.permutation(len(pool))
        shuffled_pool = pool[idx]
        
        selected = []
        selected_arr = []
        
        for v in shuffled_pool:
            # Check if v can be added
            # Condition: min distance to existing selected vectors >= sqrt(8)
            # dist_sq >= 8
            
            if not selected_arr:
                selected.append(v)
                selected_arr = [v]
                continue
                
            diffs = np.array(selected_arr) - v
            dists_sq = np.sum(diffs**2, axis=1)
            min_dist_sq = np.min(dists_sq)
            
            if min_dist_sq >= 8:
                selected.append(v)
                selected_arr.append(v)
                
        if len(selected) > len(best_set):
            best_set = selected
            
        # Optional: Try to add more from the remainder?
        # Or just rely on multiple restarts.
        
    # Convert best_set to array
    if not best_set:
        # Fallback to known valid set (E8 + A3) if greedy failed or timed out too early
        # Reconstruct the 252 set
        e8_vectors = []
        for i in range(8):
            for j in range(i + 1, 8):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(8, dtype=np.int64)
                        v[i] = 2 * si
                        v[j] = 2 * sj
                        e8_vectors.append(v)
        for bits in range(256):
            v = np.zeros(8, dtype=np.int64)
            minus_count = 0
            for i in range(8):
                if (bits >> i) & 1:
                    v[i] = -1
                    minus_count += 1
                else:
                    v[i] = 1
            if minus_count % 2 == 0:
                e8_vectors.append(v)
                
        a3_vectors = []
        for i in range(3):
            for j in range(i + 1, 3):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(3, dtype=np.int64)
                        v[i] = 2 * si
                        v[j] = 2 * sj
                        a3_vectors.append(v)
                        
        final_list = []
        for v in e8_vectors:
            full_v = np.zeros(D, dtype=np.int64)
            full_v[:8] = v
            final_list.append(full_v)
        for v in a3_vectors:
            full_v = np.zeros(D, dtype=np.int64)
            full_v[8:] = v
            final_list.append(full_v)
            
        return np.array(final_list, dtype=np.int64)

    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
