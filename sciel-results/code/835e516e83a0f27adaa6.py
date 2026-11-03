import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Generate candidates with squared norm 4
    candidates = []
    
    # Type 1: One 2, rest 0 (squared norm 4)
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s, rest 0 (squared norm 4)
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict matrix: 1 if dot > 2 (and i != j)
    # Note: M[i,i] = 4, which is > 2, but we don't conflict with ourselves.
    Conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    best_set = []
    
    def run_greedy(seed):
        if time.time() > deadline:
            return []
            
        rng = np.random.default_rng(seed)
        active = np.ones(N, dtype=bool)
        deg = Conflict.sum(axis=1).astype(np.int64)
        selected = []
        
        while active.any():
            if time.time() > deadline:
                break
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
                
            active_deg = deg[active_indices]
            min_deg = np.min(active_deg)
            candidates_for_pick = active_indices[active_deg == min_deg]
            idx = rng.choice(candidates_for_pick)
            
            selected.append(idx)
            
            # Remove idx and its neighbors
            neighbors_mask = Conflict[idx].astype(bool)
            removed_mask = np.zeros(N, dtype=bool)
            removed_mask[idx] = True
            removed_mask[neighbors_mask & active] = True
            
            removed_indices = np.where(removed_mask)[0]
            
            # Update degrees: subtract conflicts with removed nodes
            if len(removed_indices) > 0:
                removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                deg[active] -= removal_counts[active]
                
            active[removed_mask] = False
            
        return selected

    # Run with several seeds
    best_size = 0
    best_set = []
    
    # Precompute for local search: for each node, its neighbors
    neighbors = [np.where(Conflict[i] == 1)[0] for i in range(N)]
    
    def local_search(selected_indices, max_iters=20):
        if time.time() > deadline:
            return selected_indices
            
        selected_set = set(selected_indices)
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            indices_list = list(selected_set)
            rng_ls = np.random.default_rng(iters)
            rng_ls.shuffle(indices_list)
            
            for i in indices_list:
                if time.time() > deadline:
                    return list(selected_set)
                
                # Find neighbors of i not in selected_set
                nb_i = neighbors[i]
                valid_js = [j for j in nb_i if j not in selected_set]
                
                if len(valid_js) < 2:
                    continue
                
                # For each j, check if it conflicts with any k in S \ {i}
                # Use vectorized check: Conflict[j, selected_indices] excluding i
                sel_arr = np.array([k for k in selected_set if k != i])
                if len(sel_arr) == 0:
                    compatible_with_rest = valid_js
                else:
                    # Conflict[j, sel_arr] for j in valid_js
                    # Shape: (len(valid_js), len(sel_arr))
                    conflicts_matrix = Conflict[np.array(valid_js)[:, None], sel_arr[None, :]]
                    # j is compatible if no conflicts
                    compat_mask = (conflicts_matrix.sum(axis=1) == 0)
                    compatible_with_rest = [valid_js[j] for j in range(len(valid_js)) if compat_mask[j]]
                
                if len(compatible_with_rest) < 2:
                    continue
                
                # Find a compatible pair in compatible_with_rest
                cwr = compatible_with_rest
                found_pair = False
                for p in range(len(cwr)):
                    if found_pair:
                        break
                    j1 = cwr[p]
                    for q in range(p + 1, len(cwr)):
                        j2 = cwr[q]
                        if not Conflict[j1, j2]:
                            selected_set.remove(i)
                            selected_set.add(j1)
                            selected_set.add(j2)
                            improved = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                
                if found_pair:
                    break
                
        return list(selected_set)

    # Run greedy with multiple seeds
    for s in range(200):
        if time.time() > deadline:
            break
        
        res = run_greedy(s)
        if not res:
            continue
            
        # Apply local search
        res_improved = local_search(res)
        
        if len(res_improved) > best_size:
            best_size = len(res_improved)
            best_set = res_improved
            
    if not best_set:
        best_set = list(range(22))
        
    return candidates[best_set].astype(np.int64)
