import numpy as np
import itertools

D = 11

def construct():
    # Generate candidates with norm squared 8
    candidates_8 = []
    
    # Type A: Two entries +/-2
    for i, j in itertools.combinations(range(D), 2):
        for s1 in (1, -1):
            for s2 in (1, -1):
                v = [0] * D
                v[i] = 2 * s1
                v[j] = 2 * s2
                candidates_8.append(v)
                
    # Type B: One entry +/-2, four entries +/-1
    for i in range(D):
        for j, k, l, m in itertools.combinations(range(D), 4):
            if i in (j, k, l, m): continue
            # Actually combinations from range(D) doesn't exclude i automatically
            # Better: choose 5 indices, one is i, other 4 are j,k,l,m
            pass # This logic is messy. Let's do:
            
    # Refined Type B generation
    candidates_8 = []
    # Type A: 2, 2
    for i, j in itertools.combinations(range(D), 2):
        for s1 in (1, -1):
            for s2 in (1, -1):
                v = [0] * D
                v[i] = 2 * s1
                v[j] = 2 * s2
                candidates_8.append(v)
                
    # Type B: 2, 1, 1, 1, 1
    # Choose 5 distinct indices
    for idxs in itertools.combinations(range(D), 5):
        for i_pos in range(5):
            i = idxs[i_pos]
            others = [x for x in idxs if x != i]
            for s2 in (1, -1):
                for s1s in itertools.product([1, -1], repeat=4):
                    v = [0] * D
                    v[i] = 2 * s2
                    for k, idx in enumerate(others):
                        v[idx] = s1s[k]
                    candidates_8.append(v)
                    
    # Type C: 1, 1, 1, 1, 1, 1, 1, 1
    for idxs in itertools.combinations(range(D), 8):
        for signs in itertools.product([1, -1], repeat=8):
            v = [0] * D
            for k, idx in enumerate(idxs):
                v[idx] = signs[k]
            candidates_8.append(v)
            
    candidates_8 = np.array(candidates_8, dtype=np.int64)
    
    # Generate candidates with norm squared 4
    candidates_4 = []
    # Type A4: 2
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            candidates_4.append(v)
    # Type B4: 1, 1, 1, 1
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, idx in enumerate(idxs):
                v[idx] = signs[k]
            candidates_4.append(v)
            
    candidates_4 = np.array(candidates_4, dtype=np.int64)

    # Greedy Selection with Random Restarts
    best_count = 0
    best_sel_4 = []
    best_sel_8 = []
    
    # Shuffle candidates for random restarts
    np.random.seed(42)
    
    for restart in range(10):
        sel_4 = []
        sel_8 = []
        
        # Shuffle
        idx_4 = np.random.permutation(len(candidates_4))
        idx_8 = np.random.permutation(len(candidates_8))
        
        # Interleave or process separately?
        # Process 8s first? Or mix?
        # Let's process all candidates in a random order
        # But we need to know the type for constraints.
        
        # Create a list of (type, vector)
        # To save memory, we can just iterate through shuffled indices of both arrays
        # and decide which one to pick next based on random priority?
        # Simpler: Shuffle 4s and 8s separately, then merge with random weights?
        # Or just add all 8s then all 4s?
        # Adding 8s first might block 4s.
        # Let's try: Randomly pick from 4s or 8s at each step?
        # That's complex.
        
        # Strategy: 
        # 1. Shuffle 8s. Greedily add 8s.
        # 2. Shuffle 4s. Greedily add 4s that fit with existing 8s and 4s.
        
        # Restart 1: 8s first
        sel_8_arr = []
        for idx in idx_8:
            v = candidates_8[idx]
            if not sel_8_arr:
                sel_8_arr.append(v)
                continue
            sel_8_np = np.array(sel_8_arr)
            dots = sel_8_np @ v
            if np.all(dots <= 4):
                sel_8_arr.append(v)
                
        sel_4_arr = []
        for idx in idx_4:
            v = candidates_4[idx]
            # Check against sel_8
            if sel_8_arr:
                dots_8 = np.array(sel_8_arr) @ v
                if not np.all(dots_8 <= 2):
                    continue
            # Check against sel_4
            if sel_4_arr:
                dots_4 = np.array(sel_4_arr) @ v
                if not np.all(dots_4 <= 0):
                    continue
            
            sel_4_arr.append(v)
            
        count = len(sel_8_arr) + len(sel_4_arr)
        if count > best_count:
            best_count = count
            best_sel_4 = sel_4_arr
            best_sel_8 = sel_8_arr
            
    # Combine results
    all_vectors = best_sel_8 + best_sel_4
    return np.array(all_vectors, dtype=np.int64)
