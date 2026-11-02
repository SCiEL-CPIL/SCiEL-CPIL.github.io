import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
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

    # Helper: Local Search with multiple move types
    def local_search(selected_indices, max_iters=5000):
        if time.time() > deadline:
            return selected_indices
            
        S = set(selected_indices)
        conf_counts = np.zeros(N, dtype=np.int64)
        for i in S:
            conf_counts += Conflict[i]
        
        in_S = np.zeros(N, dtype=bool)
        in_S[list(S)] = True
        
        improved = True
        iter_count = 0
        
        while improved and time.time() < deadline and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            rng = np.random.default_rng(iter_count * 7919 + 13)
            s_list = list(S)
            rng.shuffle(s_list)
            
            # Phase 1: 1-for-1 swap
            for i in s_list:
                if time.time() > deadline:
                    return list(S)
                # Find j not in S with conf_counts[j] == Conflict[i, j] (0 conflicts with S\{i})
                # and j doesn't conflict with i? No, we're replacing i with j.
                # After removing i, j must have 0 conflicts with S\{i}
                valid_js = np.where((~in_S) & (conf_counts == Conflict[i]))[0]
                if len(valid_js) > 0:
                    j = valid_js[0]
                    # Verify j doesn't conflict with itself (trivially true)
                    S.discard(i)
                    S.add(j)
                    in_S[i] = False
                    in_S[j] = True
                    conf_counts -= Conflict[i]
                    conf_counts += Conflict[j]
                    improved = True
                    break
            
            # Phase 2: 1-for-2 swap
            if not improved:
                for i in s_list:
                    if time.time() > deadline:
                        return list(S)
                    if i not in S:
                        continue
                    valid_js = np.where((~in_S) & (conf_counts == Conflict[i]))[0]
                    if len(valid_js) >= 2:
                        if len(valid_js) > 100:
                            sample_idx = rng.choice(len(valid_js), size=100, replace=False)
                            valid_js_check = valid_js[sample_idx]
                        else:
                            valid_js_check = valid_js
                        
                        found_pair = False
                        for p in range(len(valid_js_check)):
                            if found_pair:
                                break
                            j1 = valid_js_check[p]
                            conflicts_with_j1 = Conflict[j1, valid_js_check]
                            compatible_qs = np.where(conflicts_with_j1 == 0)[0]
                            compatible_qs = compatible_qs[compatible_qs != p]
                            if len(compatible_qs) > 0:
                                j2 = valid_js_check[compatible_qs[0]]
                                S.discard(i)
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
            
            # Phase 3: 2-for-1 removal
            if not improved:
                s_list2 = list(S)
                rng2 = np.random.default_rng(iter_count * 104729 + 7)
                rng2.shuffle(s_list2)
                
                for idx_a in range(min(len(s_list2), 200)):
                    if time.time() > deadline:
                        return list(S)
                    i1 = s_list2[idx_a]
                    if i1 not in S:
                        continue
                    
                    for idx_b in range(idx_a + 1, min(len(s_list2), idx_a + 20)):
                        if time.time() > deadline:
                            return list(S)
                        i2 = s_list2[idx_b]
                        if i2 not in S:
                            continue
                        
                        target = Conflict[i1] + Conflict[i2]
                        valid_js2 = np.where((~in_S) & (conf_counts == target))[0]
                        
                        if len(valid_js2) >= 1:
                            j = valid_js2[0]
                            S.discard(i1)
                            S.discard(i2)
                            S.add(j)
                            in_S[i1] = False
                            in_S[i2] = False
                            in_S[j] = True
                            conf_counts -= Conflict[i1]
                            conf_counts -= Conflict[i2]
                            conf_counts += Conflict[j]
                            improved = True
                            break
                    if improved:
                        break
            
        return list(S)

    # Main loop: Try multiple seeds
    num_seeds = 500
    for s in range(num_seeds):
        if time.time() > deadline:
            break
        
        res = run_greedy(s)
        if not res:
            continue
            
        res_improved = local_search(res)
        
        if len(res_improved) > best_size:
            best_size = len(res_improved)
            best_set = res_improved

    if not best_set:
        best_set = list(range(22))
        
    return candidates[best_set].astype(np.int64)
