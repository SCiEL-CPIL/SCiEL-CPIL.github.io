import time
import numpy as np
import random

N = 17
MIN_DIST = 6
MAX_WORD = 1 << N


def _get_bad_diffs():
    """
    Precompute all differences d such that popcount(d) < MIN_DIST.
    """
    bad_diffs = []
    for d in range(MAX_WORD):
        if bin(d).count('1') < MIN_DIST:
            bad_diffs.append(d)
    return np.array(bad_diffs, dtype=np.uint32)


def construct():
    start_time = time.time()
    
    bad_diffs = _get_bad_diffs()
    num_bad = len(bad_diffs)
    
    best_code = []
    best_size = 0
    max_restarts = 5000 # Increase restarts due to faster inner loop
    
    for restart in range(max_restarts):
        if time.time() - start_time > 55:
            break
        
        # Initialize data structures
        is_avail = np.ones(MAX_WORD, dtype=bool)
        avail_list = list(range(MAX_WORD))
        pos_map = {i: i for i in range(MAX_WORD)}
        code = []
        
        while True:
            if time.time() - start_time > 55.5:
                break
            
            if not avail_list:
                break
            
            # Sample candidates
            # Adaptive sample size: larger when more available, smaller when few?
            # Or fixed moderate size.
            # Let's try 200 for better heuristic quality.
            sample_size = min(200, len(avail_list))
            
            if sample_size == 0:
                break
            
            # random.sample is faster than np.random.choice for lists?
            # np.random.choice on a list converts to array first? 
            # Let's use random.sample for the list.
            samples_list = random.sample(avail_list, sample_size)
            samples = np.array(samples_list, dtype=np.uint32)
            
            # Calculate degrees
            # (S, 1) XOR (1, B)
            neighbors = samples.reshape(-1, 1) ^ bad_diffs.reshape(1, -1)
            
            # Check availability
            # is_avail is bool array. Indexing with neighbors (S, B) array.
            neighbor_avail = is_avail[neighbors]
            
            # Count available neighbors
            degrees = neighbor_avail.sum(axis=1)
            
            # Pick min degree
            min_deg_idx = np.argmin(degrees)
            chosen_u = int(samples[min_deg_idx])
            
            code.append(chosen_u)
            
            # Update structures
            # 1. Update is_avail
            # Mark chosen_u and its neighbors as unavailable
            # Note: chosen_u is u ^ 0, so it's in bad_diffs.
            chosen_neighbors = chosen_u ^ bad_diffs
            is_avail[chosen_neighbors] = False
            
            # 2. Update avail_list and pos_map
            # We need to remove chosen_u and all chosen_neighbors that are still in avail_list
            # We can just iterate over chosen_neighbors
            for v in chosen_neighbors:
                v_int = int(v)
                if v_int in pos_map:
                    # Remove from list and map
                    idx = pos_map[v_int]
                    last_val = avail_list[-1]
                    avail_list[idx] = last_val
                    pos_map[last_val] = idx
                    avail_list.pop()
                    del pos_map[v_int]
            
            # Optimization: If the list is getting very small, we can just take all?
            # No, greedy is fine.
            
        if len(code) > best_size:
            best_size = len(code)
            best_code = code.copy()
            
    return best_code
