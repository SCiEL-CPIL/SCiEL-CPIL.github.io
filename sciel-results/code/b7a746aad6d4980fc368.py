# EVOLVE-BLOCK-START
import numpy as np
import time

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 580
    
    # All ±1 vectors in 11 dimensions: 2^11 = 2048
    all_vecs = np.zeros((2**d, d), dtype=np.int64)
    for v in range(2**d):
        for i in range(d):
            all_vecs[v, i] = 1 if (v >> i) & 1 else -1
    
    norm2 = 11  # All have norm² = 11
    
    best_points = np.array([np.ones(d), -np.ones(d)], dtype=np.int64)
    best_count = 2
    
    n_trials = 500
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        
        idx = np.random.permutation(2**d)
        shuffled = all_vecs[idx]
        
        selected = []
        selected_arr = []
        
        for vec in shuffled:
            if time.time() > deadline:
                break
            if not selected:
                selected.append(vec)
                selected_arr.append(vec)
                continue
            # Vectorized distance check
            # selected_arr is list of 1D arrays
            sel_stack = np.array(selected_arr, dtype=np.int64)  # (m, 11)
            diffs = sel_stack - vec  # (m, 11)
            dist2s = np.sum(diffs**2, axis=1)  # (m,)
            if np.all(dist2s >= norm2):
                selected.append(vec)
                selected_arr.append(vec)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_points = np.array(selected, dtype=np.int64)
    
    # Validate
    if best_count >= 2:
        pts = best_points
        norms = np.sum(pts**2, axis=1)
        max_norm = np.max(norms)
        min_dist = np.inf
        k = len(pts)
        for i in range(k):
            for j in range(i+1, k):
                diff = pts[i] - pts[j]
                d2 = np.sum(diff**2)
                if d2 < min_dist:
                    min_dist = d2
        if min_dist >= max_norm:
            return pts
    
    return np.array([np.ones(d), -np.ones(d)], dtype=np.int64)

# EVOLVE-BLOCK-END
