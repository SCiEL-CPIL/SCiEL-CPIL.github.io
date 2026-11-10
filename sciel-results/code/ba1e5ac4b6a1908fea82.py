"""Search-based construction for two Sidon sets with disjoint nonzero differences."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_size = 0
    best_a = [1]
    best_b = [2]
    
    # Precompute possible differences? No, dynamic is better for speed in numpy
    
    def _check_sidon(arr):
        if len(arr) < 2:
            return True
        a = np.array(arr)
        # Vectorized sum check
        sums = a[:, None] + a[None, :]
        return len(np.unique(sums)) == len(arr) * len(arr)
        
    def _check_diff_disjoint(a_list, b_list):
        if not a_list or not b_list:
            return True
        a = np.array(a_list)
        b = np.array(b_list)
        da = np.unique(a[:, None] - a[None, :])
        db = np.unique(b[:, None] - b[None, :])
        # Filter out 0
        da = da[da != 0]
        db = db[db != 0]
        return len(np.intersect1d(da, db)) == 0

    def _greedy(restart_seed):
        random.seed(restart_seed)
        a = []
        b = []
        
        # Shuffle candidates to break symmetry
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        # Initial seed
        if candidates:
            a.append(candidates[0])
            candidates.pop(0)
        if candidates:
            b.append(candidates[0])
            candidates.pop(0)
            
        # Greedy loop
        for x in candidates:
            if time.time() > deadline:
                return a, b
            
            # Try adding to A
            new_a = a + [x]
            if _check_sidon(new_a) and _check_diff_disjoint(new_a, b):
                a = new_a
                continue
                
            # Try adding to B
            new_b = b + [x]
            if _check_sidon(new_b) and _check_diff_disjoint(a, new_b):
                b = new_b
                continue
                
            # If neither, skip x (or we could backtrack, but greedy is faster)
            
        return a, b

    # Run multiple restarts with different seeds
    # Start with a few deterministic-ish seeds, then random
    seeds = [42, 123, 999, 31415, 27182]
    
    # Add some random seeds if time permits
    while time.time() < deadline - 5.0:
        if not seeds or time.time() > deadline - 5.0:
            seed = random.randint(1, 1000000)
        else:
            seed = seeds.pop(0)
            
        a, b = _greedy(seed)
        
        if len(a) + len(b) > best_size:
            best_size = len(a) + len(b)
            best_a = a
            best_b = b
            
        # If we've reached the theoretical max (unlikely but possible), break
        if best_size >= 64 + 64: 
            break

    return [best_a, best_b]
# EVOLVE-BLOCK-END
