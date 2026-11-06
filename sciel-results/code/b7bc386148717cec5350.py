"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    """
    Construct a binary error-correcting code of length 18 with minimum distance 6.
    Uses randomized greedy construction with bit-parallel conflict masks and local search.
    """
    N = 18
    MIN_DIST = 6
    DEADLINE = 85.0  # Leave 5s safety margin for 90s limit
    
    start_time = time.time()
    
    size = 1 << N
    
    # Precompute all masks of weight < MIN_DIST (weight 0 to 5)
    # There are sum_{i=0}^{5} C(18, i) = 1 + 18 + 153 + 816 + 3060 + 8568 = 12616 such masks
    masks_list = []
    for i in range(size):
        if i.bit_count() < MIN_DIST:
            masks_list.append(i)
    masks = np.array(masks_list, dtype=np.int64)
    
    # Precompute candidate list
    candidates = np.arange(1, size, dtype=np.int64)
    
    best_code = []
    best_len = 0
    
    # Helper to run a greedy pass given an order of candidates
    def greedy_pass(order):
        forbidden = np.zeros(size, dtype=bool)
        code = [0]
        
        # Mark neighbors of 0
        neighbors = (0 ^ masks) % size
        forbidden[neighbors] = True
        
        for c in order:
            if time.time() - start_time > DEADLINE:
                break
            c_int = int(c)
            if not forbidden[c_int]:
                code.append(c_int)
                # Mark neighbors of c
                neighbors = (c_int ^ masks) % size
                forbidden[neighbors] = True
        
        return code

    # Helper to perform local search on a code
    def local_search(code):
        if len(code) <= 1:
            return code
            
        improved = True
        max_iter = 100 
        iter_count = 0
        
        while improved and iter_count < max_iter and (time.time() - start_time < DEADLINE - 2.0):
            improved = False
            iter_count += 1
            
            idx = random.randint(0, len(code) - 1)
            removed = code[idx]
            
            new_code = code[:idx] + code[idx+1:]
            
            new_forbidden = np.zeros(size, dtype=bool)
            for c in new_code:
                neighbors = (c ^ masks) % size
                new_forbidden[neighbors] = True
            
            neighbors_of_removed = (removed ^ masks) % size
            valid_neighbors_mask = ~new_forbidden[neighbors_of_removed]
            valid_neighbors = neighbors_of_removed[valid_neighbors_mask]
            
            code_set = set(new_code)
            candidates_to_add = [c for c in valid_neighbors if c not in code_set]
            
            if candidates_to_add:
                new_c = random.choice(candidates_to_add)
                new_code.append(new_c)
                code = new_code
                improved = True
            
        return code

    # Main Loop: Randomized Greedy Restarts
    
    # First, try the standard greedy (sorted by index) as a baseline
    code = greedy_pass(candidates)
    if len(code) > best_len:
        best_code = code
        best_len = len(code)
        
    # Then, try randomized orders
    while time.time() - start_time < DEADLINE - 3.0:
        # Shuffle candidates
        order = np.random.permutation(candidates)
        
        code = greedy_pass(order)
        
        if len(code) > best_len:
            best_len = len(code)
            best_code = code[:]
            # If we found a new best, try local search on it
            if time.time() - start_time < DEADLINE - 5.0:
                improved_code = local_search(best_code)
                if len(improved_code) > best_len:
                    best_len = len(improved_code)
                    best_code = improved_code[:]
                    
    return best_code

# EVOLVE-BLOCK-END
