import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 590
    
    # Generate all vectors with exactly four 1s (and rest 0)
    # Norm squared = 4
    # We need min dist sq >= 4 => u.v <= 2
    
    candidates = []
    for indices in itertools.combinations(range(d), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for idx, s in zip(indices, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n_cands = len(candidates)
    
    # Precompute dot product matrix? 5280 x 5280 is ~28M entries.
    # 28M * 8 bytes = 224 MB. Might be too much for memory or slow to compute.
    # Let's compute on the fly or in blocks.
    
    best_set = []
    best_k = 0
    
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        order = np.random.permutation(n_cands)
        selected = []
        
        # To speed up, we can maintain a set of "forbidden" candidates?
        # Or just check against selected.
        
        for idx in order:
            if time.time() > deadline - 5:
                break
                
            v = candidates[idx]
            
            # Check against all selected
            # v . u <= 2 for all u in selected
            # If selected is empty, add.
            if not selected:
                selected.append(idx)
                continue
                
            # Vectorized check
            # selected_cands = candidates[selected]
            # dots = v @ selected_cands.T
            # if np.max(dots) <= 2:
            
            # To avoid creating large arrays repeatedly, we can check in chunks
            # or just use a loop if selected is small.
            # If selected grows large, vectorized is better.
            
            # Let's just do it. 5280 * 1000 checks is fine.
            ok = True
            # Optimization: if we store selected vectors in a list, 
            # we can compute dot products.
            
            # Actually, let's just use numpy
            if len(selected) > 0:
                sel_cands = candidates[selected]
                dots = v @ sel_cands.T
                if np.max(dots) > 2:
                    ok = False
            
            if ok:
                selected.append(idx)
                
        if len(selected) > best_k:
            best_k = len(selected)
            best_set = selected.copy()
            
    # Construct final array
    if best_set:
        result = candidates[best_set]
    else:
        # Fallback to known valid set (norm 2)
        base_points = []
        for i in range(d):
            for j in range(i + 1, d):
                base_points.append(np.array([1 if k==i else -1 if k==j else 0 for k in range(d)], dtype=np.int64))
                base_points.append(np.array([-1 if k==i else 1 if k==j else 0 for k in range(d)], dtype=np.int64))
                base_points.append(np.array([1 if k==i else 1 if k==j else 0 for k in range(d)], dtype=np.int64))
                base_points.append(np.array([-1 if k==i else -1 if k==j else 0 for k in range(d)], dtype=np.int64))
        result = np.array(base_points, dtype=np.int64)
        
    return result
