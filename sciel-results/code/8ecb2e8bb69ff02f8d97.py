import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 20s margin
    
    best_vectors = []
    best_score = 0
    
    # We know S_max=2 gives 220.
    # Try S_max = 3, 4, 5.
    # Also try S_max=2 with better greedy? No, it's full.
    
    for S_max in [3, 4, 5]:
        if time.time() > deadline:
            break
            
        candidates = generate_candidates(D, S_max)
        if not candidates:
            continue
            
        candidates = np.array(candidates, dtype=np.int64)
        
        # Try 3 random shuffles
        for trial in range(3):
            if time.time() > deadline:
                break
                
            cand_copy = candidates.copy()
            if trial > 0:
                np.random.shuffle(cand_copy)
            
            selected_indices = []
            selected_vecs = []
            
            for v in cand_copy:
                # Check compatibility with selected_vecs
                if not selected_vecs:
                    selected_indices.append(v)
                    selected_vecs.append(v)
                    continue
                
                sel_mat = np.array(selected_vecs)
                diffs = sel_mat - v
                dist_sq = np.sum(diffs * diffs, axis=1)
                if np.min(dist_sq) >= S_max:
                    selected_indices.append(v)
                    selected_vecs.append(v)
                    
            if len(selected_vecs) > best_score:
                best_score = len(selected_vecs)
                best_vectors = selected_vecs
                
    # If no improvement, return seed
    if best_score < 220:
         # Fallback to seed
         vectors = []
         for i, j in itertools.combinations(range(D), 2):
             for si in (1, -1):
                 for sj in (1, -1):
                     v = [0] * D
                     v[i], v[j] = si, sj
                     vectors.append(v)
         return np.array(vectors, dtype=np.int64)
         
    return np.array(best_vectors, dtype=np.int64)

def generate_candidates(d, max_sq_norm):
    candidates = []
    max_val = int(np.sqrt(max_sq_norm)) + 1
    
    for k in range(1, d+1):
        for indices in itertools.combinations(range(d), k):
            def gen_vals(k, remaining_norm, current_vals):
                if k == 0:
                    if 0 < sum(x*x for x in current_vals) <= max_sq_norm:
                        yield current_vals
                    return
                for v in range(-max_val, max_val + 1):
                    if v == 0: continue
                    new_norm = remaining_norm - v*v
                    if new_norm < 0: continue
                    if new_norm < k: continue
                    yield from gen_vals(k-1, new_norm, current_vals + [v])
            
            for vals in gen_vals(k, max_sq_norm, []):
                vec = [0]*d
                for idx, val in zip(indices, vals):
                    vec[idx] = val
                candidates.append(vec)
                
    return candidates
