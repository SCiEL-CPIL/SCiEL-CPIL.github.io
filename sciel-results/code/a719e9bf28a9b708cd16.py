import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate candidates with norm squared <= 3
    # This covers norms 1, sqrt(2), sqrt(3)
    candidates = []
    
    # Norm 1: (1,0,0), (-1,0,0), ...
    for i in range(D):
        for s in (1, -1):
            v = [0]*D
            v[i] = s
            candidates.append(v)
            
    # Norm sqrt(2): (1,1,0), (1,-1,0), ...
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0]*D
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Norm sqrt(3): (1,1,1), (1,1,-1), ...
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0]*D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
                    
    candidates = np.array(candidates, dtype=np.int64)
    
    best_set = []
    best_score = 0
    
    # Try different R^2 values
    # We must ensure max norm in set <= R
    # And min distance >= R
    
    # Heuristic: Greedy with random restarts
    num_restarts = 10
    for r2 in [2, 3]: # Try R=sqrt(2) and R=sqrt(3)
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Filter candidates to those with norm^2 <= r2
            norms_sq = np.sum(candidates**2, axis=1)
            valid_cands = candidates[norms_sq <= r2]
            
            # Shuffle for randomization
            np.random.shuffle(valid_cands)
            
            S = []
            S_arr = np.empty((0, D), dtype=np.int64)
            
            # To speed up, we can maintain a list of available candidates
            # But checking distance to all in S is O(|S| * |C|)
            # |C| ~ 1500, |S| max ~ 500.
            # 1500 * 500 * 11 ops = ~8M ops per restart.
            # 10 restarts * 2 R values = 20 * 8M = 160M ops.
            # Should be fast enough in Python/NumPy? Maybe tight.
            
            # Optimization: Only check against S.
            # We can remove candidates that are too close to the newly added vector.
            
            available = list(valid_cands)
            
            while available:
                if time.time() > deadline:
                    break
                    
                # Pick a candidate. 
                # Strategy: Pick the one that conflicts with the fewest other available candidates?
                # That's expensive. 
                # Just pick the first one in the shuffled list.
                v = available[0]
                
                # Check distance to all in S
                # dist_sq = ||v - u||^2 for u in S
                # If S is empty, ok.
                if len(S) == 0:
                    S.append(v)
                    S_arr = np.vstack([S_arr, v]) if S_arr.size > 0 else v.reshape(1, D)
                    # Remove from available all w such that ||w-v||^2 < r2
                    # dist_sq = ||w||^2 + ||v||^2 - 2 w.v
                    # We want dist_sq >= r2
                    # So remove if dist_sq < r2
                    dists_sq = np.sum((available - v)**2, axis=1)
                    mask = dists_sq >= r2
                    available = [available[i] for i in range(len(available)) if mask[i]]
                else:
                    dists_sq = np.sum((S_arr - v)**2, axis=1)
                    if np.min(dists_sq) >= r2:
                        S.append(v)
                        S_arr = np.vstack([S_arr, v])
                        dists_sq_avail = np.sum((available - v)**2, axis=1)
                        mask = dists_sq_avail >= r2
                        available = [available[i] for i in range(len(available)) if mask[i]]
                    else:
                        # v is too close to some u in S. Skip it.
                        # We don't remove it from available immediately, 
                        # but it will fail again. 
                        # Better to remove it? No, it might be compatible with others?
                        # No, if it's incompatible with S, it can't be added.
                        # So we can just pop it.
                        available.pop(0)
                        continue
            
            current_score = len(S)
            if current_score > best_score:
                best_score = current_score
                best_set = S
                
    if best_set:
        return np.array(best_set, dtype=np.int64)
    else:
        # Fallback to seed
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
