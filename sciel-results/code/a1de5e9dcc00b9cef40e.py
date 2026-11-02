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
    
    # Type 3: Two 1s and one 2 (1+1+4=6, too big) - skip
    # Type 4: Five 1s (5, too big) - skip
    # Only norm^2=4 vectors: (2,0,...,0) and (1,1,1,1,0,...,0)
    
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
                # Count how many active nodes lose a neighbor
                removal_counts = Conflict[removed_indices, :].sum(axis=0).astype(np.int64)
                deg[active] -= removal_counts[active]
                
            active[removed_mask] = False
            
        return selected

    # Helper: Vectorized Local Search (1-for-2 swaps and 2-for-1 removals)
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
            
            # Phase 1: Try 1-for-2 swaps (remove i, add j1, j2)
            s_list = list(S)
            rng = np.random.default_rng(iter_count * 7919 + 13)
            rng.shuffle(s_list)
            
            for i in s_list:
                if time.time() > deadline:
                    return list(S)
                
                # Find j not in S where conf_counts[j] == Conflict[i, j]
                # After removing i, j has 0 conflicts with S\{i}
                valid_js = np.where((~in_S) & (conf_counts == Conflict[i]))[0]
                
                if len(valid_js) >= 2:
                    # Find compatible pair in valid_js
                    if len(valid_js) > 50:
                        sample_idx = rng.choice(len(valid_js), size=50, replace=False)
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
            
            # Phase 2: Try 2-for-1 removal (remove i1, i2 from S, add j not in S)
            # This helps when we have redundant nodes
            if not improved:
                s_list2 = list(S)
                rng2 = np.random.default_rng(iter_count * 104729 + 7)
                rng2.shuffle(s_list2)
                
                for idx_a in range(min(len(s_list2), 100)):
                    if time.time() > deadline:
                        return list(S)
                    i1 = s_list2[idx_a]
                    if i1 not in S:
                        continue
                    
                    # Find i2 in S\{i1} such that removing both allows adding one j
                    # j must conflict with i1 and i2 but no other nodes in S\{i1,i2}
                    # conf_counts[j] - Conflict[j,i1] - Conflict[j,i2] == 0
                    # and j not in S
                    
                    # Try a few i2 candidates
                    for idx_b in range(idx_a + 1, min(len(s_list2), idx_a + 10)):
                        if time.time() > deadline:
                            return list(S)
                        i2 = s_list2[idx_b]
                        if i2 not in S:
                            continue
                        
                        # Find j: not in S, conf_counts[j] == Conflict[j,i1] + Conflict[j,i2]
                        # and Conflict[j,i1] + Conflict[j,i2] <= 2 (must be valid)
                        target = Conflict[i1] + Conflict[i2]  # conflicts j has with i1 and i2
                        valid_js2 = np.where((~in_S) & (conf_counts == target))[0]
                        
                        if len(valid_js2) >= 1:
                            j = valid_js2[0]
                            # Verify: after removing i1, i2, j has 0 conflicts
                            # conf_counts[j] - Conflict[j,i1] - Conflict[j,i2] should be 0
                            # We already checked conf_counts[j] == Conflict[j,i1] + Conflict[j,i2]
                            # So this is satisfied.
                            # Also need j to not conflict with any other node in S\{i1,i2}
                            # That's guaranteed by the equation above.
                            S.remove(i1)
                            S.remove(i2)
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

    # Main loop: Try multiple seeds with increasing effort
    num_seeds = 300
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
        # Fallback: just take first 22 (Type 1 vectors)
        best_set = list(range(22))
        
    return candidates[best_set].astype(np.int64)
