"""
Improved program: Multi-start greedy with local search for kissing number in dimension 11.
Combines the vector-level local search of the current program with the support-level
simulated annealing idea from inspiration 2 to escape local optima in the conflict graph.
"""
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    D = 11

    # Generate weight-4 vectors (norm^2=4): C(11,4)*16 = 5280
    candidates_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
    
    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)

    # Precompute norms squared
    norms_sq = np.sum(candidates.astype(np.int32) ** 2, axis=1)
    max_norm_sq = int(norms_sq.max())  # 4

    # Precompute dot product matrix
    dot = candidates.astype(np.int32) @ candidates.T.astype(np.int32)

    # Constraint: ||v_i - v_j||^2 >= max_norm_sq
    # => v_i·v_j <= (||v_i||^2 + ||v_j||^2 - max_norm_sq) / 2
    threshold = (norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - max_norm_sq) / 2.0
    compat = (dot <= threshold).astype(np.uint8)
    np.fill_diagonal(compat, 1)

    # conflict[i,j] = 1 if i and j CANNOT both be selected
    conflict = (1 - compat).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    # Precompute full degree for heuristics
    full_deg = conflict.sum(axis=1)

    def greedy_randomized(remaining_init, time_limit, seed):
        """Randomized min-degree greedy."""
        rng = np.random.default_rng(seed)
        remaining = remaining_init.copy()
        selected = []
        while time.time() < time_limit and remaining.any():
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n + 1
            min_deg = deg.min()
            if min_deg >= n:
                break
            mask = (deg == min_deg) & (remaining > 0)
            picks = np.where(mask)[0]
            if len(picks) == 0:
                break
            new_idx = rng.choice(picks)
            selected.append(new_idx)
            remaining = remaining & (1 - conflict[new_idx])
            remaining[new_idx] = 0
        return selected

    def local_search(selected_list, time_limit):
        """Local search: 1-swap and 2-swap."""
        selected = list(selected_list)
        sel_set = set(selected)
        
        # compat_with_all[j] = 1 if j is compatible with all currently selected
        compat_with_all = np.ones(n, dtype=np.uint8)
        for s in selected:
            compat_with_all = compat_with_all & compat[s]
        for s in selected:
            compat_with_all[s] = 0

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try to add a vertex (if any)
            addable = np.where(compat_with_all > 0)[0]
            if len(addable) > 0:
                # Pick the one with min full degree (least constraining)
                best_add = addable[np.argmin(full_deg[addable])]
                selected.append(best_add)
                sel_set.add(best_add)
                compat_with_all = compat_with_all & compat[best_add]
                compat_with_all[best_add] = 0
                improved = True
                continue

            # 2. Try 1-swap: Remove one, add one
            trial_indices = range(len(selected))
            if len(selected) > 100:
                trial_indices = np.random.choice(len(selected), size=min(20, len(selected)), replace=False)
            
            for si in trial_indices:
                if time.time() > time_limit:
                    break
                s = selected[si]
                
                # Compute mask of candidates compatible with all selected except s
                mask_except_s = np.ones(n, dtype=np.uint8)
                for s2 in selected:
                    if s2 != s:
                        mask_except_s = mask_except_s & compat[:, s2]
                
                mask_except_s[list(sel_set)] = 0
                
                candidates_for_swap = np.where(mask_except_s > 0)[0]
                
                if len(candidates_for_swap) > 0:
                    # Pick the best candidate (min full degree)
                    best_swap = candidates_for_swap[np.argmin(full_deg[candidates_for_swap])]
                    
                    # Perform swap
                    selected.remove(s)
                    sel_set.discard(s)
                    selected.append(best_swap)
                    sel_set.add(best_swap)
                    
                    # Recompute compat_with_all
                    compat_with_all = np.ones(n, dtype=np.uint8)
                    for s3 in selected:
                        compat_with_all = compat_with_all & compat[s3]
                    for s3 in selected:
                        compat_with_all[s3] = 0
                        
                    improved = True
                    break # Restart loop with new state

            # 3. Try 2-swap: Remove one, add two
            if not improved and time.time() < time_limit:
                trial_indices = range(len(selected))
                if len(selected) > 100:
                    trial_indices = np.random.choice(len(selected), size=min(10, len(selected)), replace=False)
                
                for si in trial_indices:
                    if time.time() > time_limit:
                        break
                    s = selected[si]
                    
                    mask_except_s = np.ones(n, dtype=np.uint8)
                    for s2 in selected:
                        if s2 != s:
                            mask_except_s = mask_except_s & compat[:, s2]
                    
                    mask_except_s[list(sel_set)] = 0
                    candidates_for_swap = np.where(mask_except_s > 0)[0]
                    
                    if len(candidates_for_swap) >= 2:
                        sorted_cands = candidates_for_swap[np.argsort(full_deg[candidates_for_swap])]
                        
                        found_pair = False
                        for i in range(min(10, len(sorted_cands))):
                            c1 = sorted_cands[i]
                            compat_with_c1 = compat[c1]
                            for j in range(i + 1, min(i + 20, len(sorted_cands))):
                                c2 = sorted_cands[j]
                                if compat_with_c1[c2]:
                                    selected.remove(s)
                                    sel_set.discard(s)
                                    selected.append(c1)
                                    sel_set.add(c1)
                                    selected.append(c2)
                                    sel_set.add(c2)
                                    
                                    compat_with_all = np.ones(n, dtype=np.uint8)
                                    for s3 in selected:
                                        compat_with_all = compat_with_all & compat[s3]
                                    for s3 in selected:
                                        compat_with_all[s3] = 0
                                        
                                    improved = True
                                    found_pair = True
                                    break
                            if found_pair:
                                break
                    if improved:
                        break

        return selected

    def simulated_annealing(selected_list, time_limit):
        """
        Simulated annealing on the vector level.
        Moves:
        1. Add a vertex if possible (always accept).
        2. Swap: Remove a random selected vertex 's', and add a random candidate 'v' 
           that is compatible with all other selected vertices.
           Delta size = 0.
           We accept based on a heuristic: prefer 'v' that has lower conflict degree 
           with the current set (excluding s) or simply random.
           To make it a proper SA, we can define an energy function.
           Let E = -len(selected).
           Move: s -> v.
           If v is compatible with all others, new size is same.
           We can try to improve "quality" by minimizing the sum of conflicts 
           of the selected set with the rest of the graph?
           Simpler: Just random swap to escape local minima of the greedy/local search.
           Acceptance: Always accept if size doesn't decrease. 
           Here size is constant. So we need a tie-breaker.
           Let's use the sum of degrees of selected vertices as a secondary objective (lower is better?).
           Or just random walk.
        """
        selected = list(selected_list)
        sel_set = set(selected)
        
        # Precompute compat_with_all for quick checks? 
        # For SA, we need to check compatibility of v with (selected - {s}).
        
        current_deg_sum = sum(full_deg[s] for s in selected)
        best_selected = selected.copy()
        best_deg_sum = current_deg_sum
        
        T = 1.0
        cooling = 0.9999
        steps = 0
        
        while time.time() < time_limit:
            steps += 1
            
            # Pick random selected vertex s to remove
            if not selected:
                break
            s = random.choice(selected)
            
            # Find candidates v compatible with all selected except s
            # mask_except_s[j] = 1 if j compatible with all s' in selected, s' != s
            mask_except_s = np.ones(n, dtype=np.uint8)
            for s2 in selected:
                if s2 != s:
                    mask_except_s = mask_except_s & compat[:, s2]
            
            # Exclude currently selected
            mask_except_s[list(sel_set)] = 0
            
            candidates_for_swap = np.where(mask_except_s > 0)[0]
            
            if len(candidates_for_swap) == 0:
                # No swap possible. Try a "kick" move? 
                # Remove s and add nothing? That decreases size. 
                # Accept with prob exp(-1/T)
                if random.random() < np.exp(-1/T):
                    selected.remove(s)
                    sel_set.discard(s)
                    current_deg_sum -= full_deg[s]
                else:
                    T *= cooling
                    continue
            
            # Pick random candidate v
            v = candidates_for_swap[random.randint(0, len(candidates_for_swap) - 1)]
            
            # Perform swap
            # Delta size = 0
            # Delta deg_sum = full_deg[v] - full_deg[s]
            delta_deg = full_deg[v] - full_deg[s]
            
            # Acceptance criterion
            # We want to minimize deg_sum (less constraining set)
            # If delta_deg < 0, accept.
            # If delta_deg > 0, accept with prob exp(-delta_deg / T)
            if delta_deg < 0 or random.random() < np.exp(-delta_deg / T):
                selected.remove(s)
                sel_set.discard(s)
                selected.append(v)
                sel_set.add(v)
                current_deg_sum += delta_deg
                
                if current_deg_sum < best_deg_sum:
                    best_deg_sum = current_deg_sum
                    best_selected = selected.copy()
            
            T *= cooling
            
            # Occasionally try to add a vertex if possible
            if steps % 10 == 0:
                # Recompute compat_with_all for add check
                compat_with_all = np.ones(n, dtype=np.uint8)
                for s3 in selected:
                    compat_with_all = compat_with_all & compat[s3]
                for s3 in selected:
                    compat_with_all[s3] = 0
                
                addable = np.where(compat_with_all > 0)[0]
                if len(addable) > 0:
                    # Pick min degree addable
                    best_add = addable[np.argmin(full_deg[addable])]
                    selected.append(best_add)
                    sel_set.add(best_add)
                    current_deg_sum += full_deg[best_add]
                    if current_deg_sum < best_deg_sum:
                        best_deg_sum = current_deg_sum
                        best_selected = selected.copy()

        return best_selected

    best_selected = []
    best_size = 0
    
    # Multi-start
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        remaining_init = np.ones(n, dtype=np.uint8)
        
        # Greedy
        sel = greedy_randomized(remaining_init, deadline - 20.0, seed=r*1000)
        
        # Local Search
        sel = local_search(sel, deadline - 10.0)
        
        # Simulated Annealing to escape local optima
        sel = simulated_annealing(sel, deadline - 5.0)
        
        if len(sel) > best_size:
            best_size = len(sel)
            best_selected = sel

    if not best_selected:
        # Fallback
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)

    # Try to add weight-2 and weight-1 vectors
    selected_vecs = candidates[best_selected]
    max_norm_sq = 4
    
    # Generate weight-2 vectors: C(11,2)*4 = 220
    w2_candidates = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            w2_candidates.append(v)
    
    # Generate weight-1 vectors: 11*2 = 22
    w1_candidates = []
    for pos in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[pos] = s
            w1_candidates.append(v)
    
    # Try to add weight-2 vectors
    extra_vecs = []
    all_selected = list(selected_vecs)
    
    for w2 in w2_candidates:
        w2_arr = np.array(w2, dtype=np.int64)
        w2_norm_sq = 2
        compatible = True
        for sv in all_selected:
            sv_arr = np.array(sv, dtype=np.int64)
            sv_norm_sq = np.sum(sv_arr ** 2)
            dist_sq = np.sum((w2_arr - sv_arr) ** 2)
            if dist_sq < max(max_norm_sq, sv_norm_sq, w2_norm_sq):
                compatible = False
                break
        if compatible:
            compatible2 = True
            for ev in extra_vecs:
                ev_arr = np.array(ev, dtype=np.int64)
                ev_norm_sq = np.sum(ev_arr ** 2)
                dist_sq = np.sum((w2_arr - ev_arr) ** 2)
                if dist_sq < max(max_norm_sq, ev_norm_sq, w2_norm_sq):
                    compatible2 = False
                    break
            if compatible2:
                extra_vecs.append(w2)
    
    # Try to add weight-1 vectors
    for w1 in w1_candidates:
        w1_arr = np.array(w1, dtype=np.int64)
        w1_norm_sq = 1
        compatible = True
        for sv in all_selected:
            sv_arr = np.array(sv, dtype=np.int64)
            sv_norm_sq = np.sum(sv_arr ** 2)
            dist_sq = np.sum((w1_arr - sv_arr) ** 2)
            if dist_sq < max(max_norm_sq, sv_norm_sq, w1_norm_sq):
                compatible = False
                break
        if compatible:
            compatible2 = True
            for ev in extra_vecs:
                ev_arr = np.array(ev, dtype=np.int64)
                ev_norm_sq = np.sum(ev_arr ** 2)
                dist_sq = np.sum((w1_arr - ev_arr) ** 2)
                if dist_sq < max(max_norm_sq, ev_norm_sq, w1_norm_sq):
                    compatible2 = False
                    break
            if compatible2:
                extra_vecs.append(w1)
    
    # Combine
    if extra_vecs:
        all_vecs = list(selected_vecs) + extra_vecs
        result = np.array(all_vecs, dtype=np.int64)
    else:
        result = selected_vecs.astype(np.int64)
    
    return result

