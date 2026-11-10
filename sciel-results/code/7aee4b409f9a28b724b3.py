# EVOLVE-BLOCK-START
import numpy as np
import time


def kissing_number11() -> np.ndarray:
    """
    Finds a large set of integer points in 11 dimensions such that the minimum
    squared distance between any pair is at least the maximum squared norm.
    
    We search for a maximum independent set in the graph of candidate vectors
    (norm squared 2 and 4) where edges connect pairs with squared distance < 4.
    This corresponds to a configuration with Max Norm Squared = 4 and
    Min Squared Distance >= 4.
    """
    d = 11
    candidates = []
    
    # Generate Norm Squared 2 vectors (D11 minimal vectors)
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
                    
    # Generate Norm Squared 4 vectors
    # Type A: One coordinate is +/-2
    for i in range(d):
        for s in [2, -2]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # Type B: Two coordinates are +/-1
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = s1
                    v[j] = s2
                    candidates.append(v)
                    
    # Type C: Four coordinates are +/-1
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                for l in range(k + 1, d):
                    for s1 in [1, -1]:
                        for s2 in [1, -1]:
                            for s3 in [1, -1]:
                                for s4 in [1, -1]:
                                    v = np.zeros(d, dtype=np.int64)
                                    v[i] = s1
                                    v[j] = s2
                                    v[k] = s3
                                    v[l] = s4
                                    candidates.append(v)
                                    
    candidates = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates)
    
    # Greedy Independent Set with multiple restarts
    best_set = []
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # Sort candidates by norm descending? 
    # Norm 4 vectors are more "spread out" potentially?
    # Let's try random shuffling for diversity.
    
    for restart in range(10000):
        if time.time() > deadline:
            break
            
        indices = np.random.permutation(n_candidates)
        S = []
        S_arr = np.empty((0, d), dtype=np.int64)
        
        for idx in indices:
            c = candidates[idx]
            
            if len(S) == 0:
                S.append(c)
                S_arr = np.array([c], dtype=np.int64)
            else:
                # Check distance to all in S
                # dist2 = sum((S_arr - c)**2, axis=1)
                # If any dist2 < 4, reject
                diffs = S_arr - c
                dists2 = np.sum(diffs * diffs, axis=1)
                
                if np.any(dists2 < 4):
                    continue
                else:
                    S.append(c)
                    S_arr = np.vstack([S_arr, c])
                    
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) > 400: # Heuristic: good enough?
                # We can try to continue to find better, but 400 is already > 220
                # Let's keep searching a bit more if time permits
                pass
                
    return np.array(best_set, dtype=np.int64)

# EVOLVE-BLOCK-END
