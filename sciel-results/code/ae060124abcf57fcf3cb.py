import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Generate candidates with squared norm 4
    candidates = []
    
    # Type 1: One 2, rest 0
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four 1s, rest 0
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
    Conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(Conflict, 0)
    
    best_set = []
    best_size = 0
    
    # Helper: Run greedy
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
            
            neighbors_mask = Conflict[idx].astype(bool)
            removed_mask = np.zeros(N, dtype=bool)
            removed_mask[idx] = True
            removed_mask[neighbors_mask & active] = True
            
            removed_indices = np.where(removed_mask)[0]
            
            if len(removed_indices) > 0:
                removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                deg[active] -= removal_counts[active]
                
            active[removed_mask] = False
            
        return selected

    # Helper: Vectorized Local Search (1-for-2 swaps)
    def local_search(selected_indices):
        if time.time() > deadline:
            return selected_indices
            
        S = set(selected_indices)
        # conf_counts[j] = number of neighbors of j in S
        conf_counts = np.zeros(N, dtype=np.int64)
        for i in S:
            conf_counts += Conflict[i]
        
        in_S = np.zeros(N, dtype=bool)
        in_S[list(S)] = True
        
        improved = True
        max_iterations = 2000
        iter_count = 0
        
        while improved and time.time() < deadline and iter_count < max_iterations:
            improved = False
            iter_count += 1
            
            s_list = list(S)
            rng = np.random.default_rng(iter_count)
            rng.shuffle(s_list)
            
            for i in s_list:
                if time.time() > deadline:
                    return list(S)
                
                # Find candidates j not in S such that conf_counts[j] == Conflict[i, j]
                # This means if we remove i, j has 0 conflicts with S\{i}
                valid_js = np.where((~in_S) & (conf_counts == Conflict[i]))[0]
                
                if len(valid_js) < 2:
                    continue
                
                # Find a pair j1, j2 in valid_js with no conflict
                found_pair = False
                
                if len(valid_js) > 50:
                    # Sample a subset
                    sample_size = min(50, len(valid_js))
                    sample_idx = np.random.default_rng(iter_count + i).choice(len(valid_js), size=sample_size, replace=False)
                    valid_js_check = valid_js[sample_idx]
                else:
                    valid_js_check = valid_js
                
                for p in range(len(valid_js_check)):
                    if found_pair:
                        break
                    j1 = valid_js_check[p]
                    conflicts_with_j1 = Conflict[j1, valid_js_check]
                    compatible_qs = np.where(conflicts_with_j1 == 0)[0]
                    compatible_qs = compatible_qs[compatible_qs != p]
                    if len(compatible_qs) > 0:
                        j2 = valid_js_check[compatible_qs[0]]
                        
                        # Perform swap: remove i, add j1, j2
                        S.remove(i)
                        S.add(j1)
                        S.add(j2)
                        
                        in_S[i] = False
                        in_S[j1] = True
                        in_S[j2] = True
                        
                        conf_counts -= Conflict[i]
                        conf_counts += Conflict[j1]
                        conf_counts += Conflict[j2]
                        
                        improved = True
                        found_pair = True
                        break
                        
        return list(S)

    # Helper: 1-for-1 swap local search
    def local_search_1for1(selected_indices):
        if time.time() > deadline:
            return selected_indices
            
        S = set(selected_indices)
        conf_counts = np.zeros(N, dtype=np.int64)
        for i in S:
            conf_counts += Conflict[i]
        
        in_S = np.zeros(N, dtype=bool)
        in_S[list(S)] = True
        
        improved = True
        max_iterations = 1000
        iter_count = 0
        
        while improved and time.time() < deadline and iter_count < max_iterations:
            improved = False
            iter_count += 1
            
            s_list = list(S)
            rng = np.random.default_rng(100000 + iter_count)
            rng.shuffle(s_list)
            
            for i in s_list:
                if time.time() > deadline:
                    return list(S)
                
                # Find j not in S with conf_counts[j] == 0 (compatible with all of S)
                # If we replace i with j, j must be compatible with S\{i}
                # conf_counts[j] - Conflict[j,i] == 0
                valid_js = np.where((~in_S) & (conf_counts == Conflict[i]))[0]
                
                if len(valid_js) == 0:
                    continue
                
                # Pick a random valid j
                j = rng.choice(valid_js)
                
                # Perform swap
                S.remove(i)
                S.add(j)
                
                in_S[i] = False
                in_S[j] = True
                
                conf_counts -= Conflict[i]
                conf_counts += Conflict[j]
                
                improved = True
                break  # Restart iteration after a swap
                
        return list(S)

    # Main loop: Try multiple seeds
    num_seeds = 500
    for s in range(num_seeds):
        if time.time() > deadline:
            break
        
        res = run_greedy(s)
        if not res:
            continue
            
        # Apply 1-for-1 local search first
        res_11 = local_search_1for1(res)
        # Then apply 1-for-2 local search
        res_improved = local_search(res_11)
        
        if len(res_improved) > best_size:
            best_size = len(res_improved)
            best_set = res_improved

    if not best_set:
        # Fallback: take first 22 vectors
        best_set = list(range(22))
        
    return candidates[best_set].astype(np.int64)
