import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Generate pool of vectors
    pool = []
    
    # Helper to add vector if not already present
    def add_vec(v):
        # v is a tuple of ints
        # Check norm squared
        n2 = sum(x*x for x in v)
        if 1 <= n2 <= 5: # We can go up to 5 or 8
            # Normalize to tuple for hashing?
            # Just append, we can deduplicate later or use set
            pool.append(np.array(v, dtype=np.int64))
            
    # Generate all vectors with small norm
    # This is a bit slow if done naively.
    # Better to generate by patterns.
    
    # Pattern: k non-zero entries
    for k in range(1, 6):
        if time.time() > deadline: break
        # Choose indices
        for idxs in itertools.combinations(range(D), k):
            # Choose values
            # Max value such that sum of squares <= 5
            # If k=1, val in 1,2
            # If k=2, vals in 1,2 such that v1^2+v2^2<=5
            # If k=3, vals in 1 such that 1+1+1<=5
            # If k=4, vals in 1
            # If k=5, vals in 1 (norm 5)
            
            # Determine allowed values for each position
            # Simple: iterate all combinations of values from {-2, -1, 1, 2}
            # and check norm
            # 4^k possibilities.
            # k=1: 4
            # k=2: 16
            # k=3: 64
            # k=4: 256
            # k=5: 1024
            
            vals_range = [-2, -1, 1, 2]
            for vals in itertools.product(vals_range, repeat=k):
                n2 = sum(v*v for v in vals)
                if n2 > 5:
                    continue
                if n2 < 1:
                    continue
                v = [0]*D
                for idx, val in zip(idxs, vals):
                    v[idx] = val
                add_vec(v)
                
    # Deduplicate
    # Use a set of tuples
    unique_pool = set()
    for v in pool:
        unique_pool.add(tuple(v))
        
    vectors = [np.array(v, dtype=np.int64) for v in unique_pool]
    
    # Greedy Search
    best_set = []
    best_size = 0
    
    # Shuffle and restart
    # Number of restarts depends on time
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        np.random.shuffle(vectors)
        
        current_set = []
        current_max_norm_sq = 0
        
        for v in vectors:
            v_norm_sq = np.dot(v, v)
            new_max_norm_sq = max(current_max_norm_sq, v_norm_sq)
            
            # Check compatibility with all in current_set
            # Condition: ||v - u||^2 >= new_max_norm_sq
            compatible = True
            for u in current_set:
                diff = v - u
                dist_sq = np.dot(diff, diff)
                if dist_sq < new_max_norm_sq:
                    compatible = False
                    break
            
            if compatible:
                current_set.append(v)
                current_max_norm_sq = new_max_norm_sq
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    return np.array(best_set, dtype=np.int64)
