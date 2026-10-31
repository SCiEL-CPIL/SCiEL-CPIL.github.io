import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate all weight-2 vectors
    w2_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                w2_vectors.append(v)
    w2_vectors = np.array(w2_vectors)
    
    # Generate all weight-4 vectors
    w4_vectors = []
    for indices in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(indices, signs):
                v[idx] = s
            w4_vectors.append(v)
    w4_vectors = np.array(w4_vectors)
    
    # Combine
    all_vectors = np.vstack([w2_vectors, w4_vectors])
    N = len(all_vectors)
    
    # Pre-compute squared norms
    # w2: norm^2 = 2
    # w4: norm^2 = 4
    norms_sq = np.zeros(N, dtype=np.int64)
    norms_sq[:len(w2_vectors)] = 2
    norms_sq[len(w2_vectors):] = 4
    
    # We want to find a subset S such that for all u, v in S:
    # d(u, v)^2 >= max(norms_sq in S)
    # If S contains any w4, max norm sq is 4. Condition: d^2 >= 4.
    # If S contains only w2, max norm sq is 2. Condition: d^2 >= 2.
    
    # Strategy: Try to find a large set satisfying d^2 >= 4.
    # If we fail to beat 220, we return the 220 w2 vectors.
    
    best_set = []
    best_score = 0
    
    # Seed solution: 220 w2 vectors
    # Check if they are valid under d^2 >= 2 (they are)
    # Score is 220.
    current_best_size = 220
    current_best_vectors = w2_vectors
    
    # Greedy search for d^2 >= 4
    # We will try a few random orderings
    
    for trial in range(20):
        if time.time() > deadline:
            break
            
        # Random shuffle
        indices = np.random.permutation(N)
        
        selected_indices = []
        selected_vectors = []
        
        # To speed up, we can just iterate
        # For each candidate, check against selected
        # If selected is empty, add
        # Else, compute min dist sq to selected
        # If min dist sq >= 4, add
        
        for idx in indices:
            if time.time() > deadline:
                break
                
            v = all_vectors[idx]
            norm_v = norms_sq[idx]
            
            if not selected_vectors:
                selected_vectors.append(v)
                selected_indices.append(idx)
                continue
                
            # Compute distances to all selected
            # selected_vectors is a list of arrays
            # Convert to array for vectorized op?
            # Or just loop?
            # N is 5500. Selected size grows to ~250.
            # 5500 * 250 * 11 ops = 15M ops per trial.
            # 20 trials = 300M ops. Might be tight in Python loops.
            # Use NumPy.
            
            sel_arr = np.array(selected_vectors)
            # dist_sq = ||v||^2 + ||s||^2 - 2 v.s
            # v.s for all s in sel_arr
            dots = sel_arr @ v
            sel_norms = np.array([norms_sq[i] for i in selected_indices])
            dist_sqs = norm_v + sel_norms - 2 * dots
            
            if np.min(dist_sqs) >= 4:
                selected_vectors.append(v)
                selected_indices.append(idx)
                
        if len(selected_vectors) > current_best_size:
            current_best_size = len(selected_vectors)
            current_best_vectors = np.array(selected_vectors)
            
    # If we found something better than 220, return it.
    # Note: The set found by greedy (d^2>=4) is valid for the task?
    # Task: min pairwise dist >= max norm.
    # If set has max norm 2 (from w4), min dist >= 2.
    # dist >= 2 implies dist^2 >= 4.
    # So yes, valid.
    
    # If we didn't find better, return the seed (220 w2 vectors)
    # The seed is valid?
    # Max norm sqrt(2). Min dist >= sqrt(2).
    # dist^2 >= 2.
    # Yes.
    
    return current_best_vectors
