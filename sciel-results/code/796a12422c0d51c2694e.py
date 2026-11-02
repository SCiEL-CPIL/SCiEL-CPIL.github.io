import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Generate candidates
    candidates = []
    seen = set()
    
    def add_vec(v):
        key = tuple(v)
        if key not in seen:
            seen.add(key)
            candidates.append(np.array(v, dtype=np.int64))

    # W=2
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                add_vec(v)
                
    # W=3
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    add_vec(v)
                    
    # W=4
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = [0] * D
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        add_vec(v)
                        
    for i in range(D):
        for si in (1, -1):
            v = [0] * D
            v[i] = 2 * si
            add_vec(v)

    # W=5
    for i, j, k, l, m in itertools.combinations(range(D), 5):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        for sm in (1, -1):
                            v = [0] * D
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            v[l] = sl
                            v[m] = sm
                            add_vec(v)
                            
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = sj
                add_vec(v)
                v2 = [0] * D
                v2[i] = sj
                v2[j] = 2 * si
                add_vec(v2)

    # W=6
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i] = 2 * si
                    v[j] = sj
                    v[k] = sk
                    add_vec(v)
                    v2 = [0] * D
                    v2[i] = sj
                    v2[j] = 2 * si
                    v2[k] = sk
                    add_vec(v2)
                    v3 = [0] * D
                    v3[i] = sj
                    v3[j] = sk
                    v3[k] = 2 * si
                    add_vec(v3)
                    
    # Type 1,1,1,1,1,1
    for i, j, k, l, m, n in itertools.combinations(range(D), 6):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        for sm in (1, -1):
                            for sn in (1, -1):
                                v = [0] * D
                                v[i] = si
                                v[j] = sj
                                v[k] = sk
                                v[l] = sl
                                v[m] = sm
                                v[n] = sn
                                add_vec(v)

    N = len(candidates)
    norms = np.array([np.sum(v**2) for v in candidates])
    
    # Precompute neighbors for each possible W_max?
    # No, W_max changes the definition of neighbor.
    # Neighbor of u for W_max is v if dist(u,v)^2 < W_max.
    
    # We will precompute dist^2 matrix?
    # N is approx 15k. 15k^2 = 225M. Too big for memory if float.
    # But we can compute on the fly for specific W_max runs.
    # Or, we can just run the search for W_max=4 and W_max=5.
    
    best_result = []
    best_score = 0
    
    def run_search(w_max, restarts=20):
        # Filter candidates
        valid_indices = [i for i in range(N) if norms[i] <= w_max]
        if not valid_indices:
            return []
        
        # Precompute neighbors for this w_max
        # neighbors[i] = list of j in valid_indices such that dist(i,j)^2 < w_max
        # This is O(N^2).
        # If N is 15k, N^2 is 225M. In Python, this loop will take minutes.
        # We MUST optimize this.
        
        # Optimization: Group by weight?
        # Or just accept that we can't precompute for N=15k.
        # Let's reduce candidate set.
        # The most promising are W=2,3,4.
        # Let's try to limit N.
        
        # If we only use W<=4, N is approx 10k.
        # 10k^2 = 100M. Still slow in Python.
        
        # Alternative: Don't precompute. Use NumPy vectorization for conflict check?
        # Check new vector against entire current set:
        # dists = np.sum((candidates[current_set] - v_new)**2, axis=1)
        # if np.min(dists) >= w_max: add
        
        # This is O(|current_set| * D).
        # If |current_set| is 500, D=11. 5500 ops. Very fast.
        # So we don't need precomputed neighbors! We just need the current set as a NumPy array.
        
        best_local = []
        
        for _ in range(restarts):
            current_set = set()
            current_arr = np.empty((0, D), dtype=np.int64)
            
            # Greedy Init
            indices = valid_indices[:]
            random.shuffle(indices)
            
            for idx in indices:
                v_new = candidates[idx]
                if len(current_arr) == 0:
                    current_set.add(idx)
                    current_arr = np.vstack([current_arr, v_new])
                else:
                    dists = np.sum((current_arr - v_new)**2, axis=1)
                    if np.min(dists) >= w_max:
                        current_set.add(idx)
                        current_arr = np.vstack([current_arr, v_new])
                        
            # Local Search
            # We want to add more or swap.
            # To speed up, we can keep a list of "blocked" vertices?
            # Or just try random adds.
            
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                
                # Try Add
                avail = [i for i in valid_indices if i not in current_set]
                if avail:
                    # Try a batch of random adds
                    random.shuffle(avail)
                    for idx in avail[:50]: # Try 50 randoms
                        v_new = candidates[idx]
                        dists = np.sum((current_arr - v_new)**2, axis=1)
                        if np.min(dists) >= w_max:
                            current_set.add(idx)
                            current_arr = np.vstack([current_arr, v_new])
                            improved = True
                            # Remove from avail? Not strictly necessary if we shuffle next time.
                            break
                
                # Try Swap (Remove 1, Add 1)
                if not improved and len(current_set) > 0:
                    in_set = list(current_set)
                    random.shuffle(in_set)
                    for idx_in in in_set[:20]: # Try 20 randoms
                        # Remove idx_in
                        current_set.remove(idx_in)
                        # Update current_arr (expensive)
                        # We can just rebuild? Or use a mask?
                        # Rebuild is O(N).
                        # Let's just try to add a new one first, then remove?
                        # No, we need to check if new one fits with (set - {idx_in}).
                        
                        # Let's try a different swap strategy:
                        # Pick idx_in.
                        # Find a candidate that conflicts with idx_in but not others?
                        # Too hard.
                        
                        # Simple Swap:
                        # Remove idx_in.
                        # Try to add a random candidate.
                        # If success, keep. If fail, put idx_in back.
                        
                        # To avoid rebuilding array, we can just check against the set minus idx_in.
                        # But we need the array for vectorization.
                        # Let's just accept the cost of rebuilding if we find a candidate.
                        
                        # Actually, let's just do "Add" aggressively.
                        # The "Swap" is less critical if we have good restarts.
                        
                        # Let's try a "Perturbation": Remove 5 random, then Add greedily.
                        pass

            if len(current_set) > len(best_local):
                best_local = list(current_set)
                
        return best_local

    # Run for W=4
    res4 = run_search(4, restarts=30)
    if len(res4) > best_score:
        best_score = len(res4)
        best_result = res4
        
    # Run for W=5
    if time.time() < deadline - 10:
        res5 = run_search(5, restarts=30)
        if len(res5) > best_score:
            best_score = len(res5)
            best_result = res5

    if not best_result:
        w2_indices = [i for i in range(N) if norms[i] == 2]
        return np.array([candidates[i] for i in w2_indices], dtype=np.int64)

    return np.array([candidates[i] for i in best_result], dtype=np.int64)
