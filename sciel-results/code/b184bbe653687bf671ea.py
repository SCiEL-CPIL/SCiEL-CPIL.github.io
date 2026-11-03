"""Combined: support-level MIS + SA + 2-for-2 polish for kissing number in dimension 11."""
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s limit
    
    D = 11
    
    # Generate all C(11,4) = 330 supports
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)  # 330
    
    # Build conflict matrix: edge if intersection size == 3
    # (intersection 4 means same support, intersection <=2 means no conflict)
    support_sets = [set(s) for s in supports]
    adj_mask = np.zeros((n_sup, n_sup), dtype=np.uint8)
    for i in range(n_sup):
        for j in range(i + 1, n_sup):
            if len(support_sets[i] & support_sets[j]) == 3:
                adj_mask[i, j] = 1
                adj_mask[j, i] = 1
    
    # Precompute adjacency lists
    adj_lists = [np.where(adj_mask[i] == 1)[0] for i in range(n_sup)]
    
    def greedy_is(seed=0):
        """Greedy independent set using minimum-degree heuristic with random tie-breaking."""
        rng = np.random.default_rng(seed)
        remaining = np.ones(n_sup, dtype=np.uint8)
        selected = np.zeros(n_sup, dtype=np.uint8)
        
        while True:
            if time.time() > deadline - 15.0:
                break
            rem_idx = np.where(remaining > 0)[0]
            if len(rem_idx) == 0:
                break
            
            # Compute degree within remaining set
            deg = (adj_mask * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n_sup + 1
            
            min_deg = deg.min()
            candidates = np.where((deg == min_deg) & (remaining > 0))[0]
            if len(candidates) == 0:
                break
            
            pick = rng.choice(candidates)
            selected[pick] = 1
            remaining[pick] = 0
            remaining[adj_mask[pick] == 1] = 0
        
        return selected
    
    def local_search_sa(selected, n_steps=50000, seed=0):
        """Simulated annealing with 1-for-k moves on the support-level problem."""
        rng = np.random.default_rng(seed)
        rand = random.Random(seed)
        
        S = selected.copy()
        current_size = int(S.sum())
        best_size = current_size
        best_S = S.copy()
        
        T = 2.0
        cooling = 0.9997
        min_T = 0.05
        
        step = 0
        while step < n_steps and time.time() < deadline - 10.0:
            step += 1
            v = rng.integers(0, n_sup)
            
            if S[v]:
                # Removal move (probabilistic)
                if rand.random() < 0.03:
                    S[v] = 0
                    current_size -= 1
                continue
            
            # v not in S: try to add it
            conflicts = np.where(adj_mask[v] & S)[0]
            n_conf = len(conflicts)
            
            if n_conf == 0:
                S[v] = 1
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_S = S.copy()
            else:
                # 1-for-n move: remove n_conf conflicting supports, add v
                delta = 1 - n_conf
                if delta >= 0:
                    S[conflicts] = 0
                    S[v] = 1
                    current_size += delta
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S.copy()
                else:
                    if T > min_T:
                        p = np.exp(delta / T)
                        if rand.random() < p:
                            S[conflicts] = 0
                            S[v] = 1
                            current_size += delta
                            if current_size > best_size:
                                best_size = current_size
                                best_S = S.copy()
            
            T = max(min_T, T * cooling)
        
        return best_S, best_size
    
    def polish_2for2(S, max_rounds=20):
        """Try all 2-for-2 moves: remove 2 supports, add 2 others."""
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 8.0:
            improved = False
            rounds += 1
            
            sel = np.where(S == 1)[0]
            k = len(sel)
            if k < 2:
                break
            
            # For each pair of selected supports, try to replace with a better pair
            n_trials = min(5000, k * k // 2)
            for _ in range(n_trials):
                if time.time() > deadline - 8.0:
                    break
                i1, i2 = np.random.default_rng().integers(0, k, size=2)
                if i1 == i2:
                    continue
                v1, v2 = sel[i1], sel[i2]
                
                # Find all supports compatible with both v1 and v2 (not conflicting)
                # and not currently selected
                compat = (adj_mask[v1] == 0) & (adj_mask[v2] == 0) & (S == 0)
                candidates = np.where(compat)[0]
                if len(candidates) < 2:
                    continue
                
                # Sample pairs from candidates
                for trial in range(min(10, len(candidates) - 1)):
                    u_idx = np.random.choice(len(candidates))
                    w_idx = np.random.choice(len(candidates) - 1)
                    if w_idx >= u_idx:
                        w_idx += 1
                    u, w = candidates[u_idx], candidates[w_idx]
                    if adj_mask[u, w] == 1:
                        continue
                    
                    # Check: removing v1, v2 and adding u, w improves
                    others = np.delete(sel, [i1, i2])
                    if len(others) > 0:
                        if np.any(adj_mask[u][others] == 1) or np.any(adj_mask[w][others] == 1):
                            continue
                    
                    S[v1] = 0
                    S[v2] = 0
                    S[u] = 1
                    S[w] = 1
                    improved = True
                    break
                if improved:
                    break
        
        return S
    
    # Main search: multiple restarts
    best_S = np.zeros(n_sup, dtype=np.uint8)
    best_size = 0
    
    # Initial greedy starts with different seeds
    n_restarts = 30
    for i in range(n_restarts):
        if time.time() > deadline - 25.0:
            break
        
        # Greedy initialization
        S_init = greedy_is(seed=1000 + i)
        init_size = int(S_init.sum())
        
        # SA refinement
        S_sa, sa_size = local_search_sa(S_init, n_steps=30000, seed=2000 + i)
        
        # Polish
        S_polished = polish_2for2(S_sa.copy(), max_rounds=10)
        pol_size = int(S_polished.sum())
        
        if pol_size > best_size:
            best_size = pol_size
            best_S = S_polished.copy()
        
        # Also try SA with longer run every few restarts
        if i % 5 == 0 and time.time() < deadline - 30.0:
            S_sa2, sa2_size = local_search_sa(S_init, n_steps=80000, seed=5000 + i)
            S_pol2 = polish_2for2(S_sa2.copy(), max_rounds=15)
            pol2_size = int(S_pol2.sum())
            if pol2_size > best_size:
                best_size = pol2_size
                best_S = S_pol2.copy()
    
    # Expand support-level solution to vectors
    # For each selected support, include all 16 sign combinations
    selected_supports = np.where(best_S == 1)[0]
    
    # Generate all weight-4 vectors for selected supports
    vectors = []
    for sup_idx in selected_supports:
        positions = supports[sup_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            vectors.append(v)
    
    # Now try to add weight-2 and weight-1 vectors
    # Condition: for a new vector v and all selected weight-4 vectors s:
    #   dist(v, s)^2 >= 4
    #   ||v||^2 + ||s||^2 - 2*v.s >= 4
    #   ||v||^2 + 4 - 2*v.s >= 4
    #   ||v||^2 - 2*v.s >= 0
    #   v.s <= ||v||^2 / 2
    
    # For weight-2 (||v||^2 = 2): v.s <= 1
    # For weight-1 (||v||^2 = 1): v.s <= 0.5, so v.s <= 0 (integer)
    
    # Build array of selected weight-4 vectors
    if len(vectors) > 0:
        wt4_arr = np.array(vectors, dtype=np.int32)
        n_wt4 = len(wt4_arr)
        
        # Generate weight-2 candidates
        wt2_candidates = []
        for positions in itertools.combinations(range(D), 2):
            for signs in itertools.product([1, -1], repeat=2):
                v = [0] * D
                for pos, s in zip(positions, signs):
                    v[pos] = s
                wt2_candidates.append(v)
        
        # Generate weight-1 candidates
        wt1_candidates = []
        for pos in range(D):
            for s in [1, -1]:
                v = [0] * D
                v[pos] = s
                wt1_candidates.append(v)
        
        # Check weight-2 candidates
        if len(wt2_candidates) > 0:
            wt2_arr = np.array(wt2_candidates, dtype=np.int32)
            # dot products with all weight-4 vectors
            dots_wt2 = wt2_arr @ wt4_arr.T  # (n_wt2, n_wt4)
            # Condition: v.s <= 1 for all s
            valid_wt2 = np.all(dots_wt2 <= 1, axis=1)
            
            # Also need weight-2 vectors to be mutually compatible
            # dist(v1, v2)^2 >= 4 for weight-2 vectors
            # ||v1||^2 + ||v2||^2 - 2*v1.v2 >= 4
            # 2 + 2 - 2*v1.v2 >= 4
            # v1.v2 <= 0
            if np.any(valid_wt2):
                valid_wt2_idx = np.where(valid_wt2)[0]
                valid_wt2_arr = wt2_arr[valid_wt2_idx]
                n_valid_wt2 = len(valid_wt2_arr)
                
                if n_valid_wt2 > 1:
                    dots_wt2_wt2 = valid_wt2_arr @ valid_wt2_arr.T
                    # Need v1.v2 <= 0 for i != j
                    wt2_compat = (dots_wt2_wt2 <= 0).astype(np.uint8)
                    np.fill_diagonal(wt2_compat, 1)
                    
                    # Find maximum independent set in wt2_compat graph
                    # Use greedy
                    remaining = np.ones(n_valid_wt2, dtype=np.uint8)
                    selected_wt2 = []
                    while remaining.any():
                        deg = (1 - wt2_compat) * remaining[np.newaxis, :]
                        deg = deg.sum(axis=1)
                        deg[remaining == 0] = n_valid_wt2 + 1
                        min_deg = deg.min()
                        if min_deg >= n_valid_wt2:
                            break
                        candidates = np.where((deg == min_deg) & (remaining > 0))[0]
                        if len(candidates) == 0:
                            break
                        pick = candidates[0]
                        selected_wt2.append(pick)
                        remaining[pick] = 0
                        remaining[1 - wt2_compat[pick]] = 0
                    
                    for idx in selected_wt2:
                        vectors.append(wt2_candidates[valid_wt2_idx[idx]])
        
        # Check weight-1 candidates
        if len(wt1_candidates) > 0:
            wt1_arr = np.array(wt1_candidates, dtype=np.int32)
            dots_wt1 = wt1_arr @ wt4_arr.T  # (n_wt1, n_wt4)
            # Condition: v.s <= 0 for all s
            valid_wt1 = np.all(dots_wt1 <= 0, axis=1)
            
            # Also need weight-1 vectors to be compatible with selected weight-2 vectors
            # and mutually compatible
            # dist(v1, v2)^2 >= 4 for weight-1 vectors
            # 1 + 1 - 2*v1.v2 >= 4
            # v1.v2 <= -1
            if np.any(valid_wt1):
                valid_wt1_idx = np.where(valid_wt1)[0]
                valid_wt1_arr = wt1_arr[valid_wt1_idx]
                n_valid_wt1 = len(valid_wt1_arr)
                
                # Check compatibility with existing selected vectors (weight-4 and weight-2)
                # For weight-1 v and weight-4 s: v.s <= 0 (already checked)
                # For weight-1 v and weight-2 s: 
                #   1 + 2 - 2*v.s >= 4 => v.s <= -0.5 => v.s <= -1
                # For weight-1 v and weight-1 s: v.s <= -1
                
                # Build full set of selected vectors so far
                selected_arr = np.array(vectors, dtype=np.int32)
                n_sel = len(selected_arr)
                
                # Check weight-1 against all selected
                if n_sel > 0:
                    dots_wt1_sel = valid_wt1_arr @ selected_arr.T
                    # For each selected vector, check the appropriate condition
                    # We need to know the norm of each selected vector
                    sel_norms_sq = np.sum(selected_arr ** 2, axis=1)
                    
                    # Condition: v.s <= (||v||^2 + ||s||^2 - 4) / 2
                    # For weight-1: ||v||^2 = 1
                    # threshold = (1 + sel_norms_sq - 4) / 2 = (sel_norms_sq - 3) / 2
                    thresholds = (sel_norms_sq - 3) / 2.0
                    # dots_wt1_sel[i, j] <= thresholds[j] for all j
                    valid_wt1_sel = np.all(dots_wt1_sel <= thresholds[np.newaxis, :], axis=1)
                else:
                    valid_wt1_sel = np.ones(n_valid_wt1, dtype=bool)
                
                # Mutual compatibility among weight-1: v.s <= -1
                if np.any(valid_wt1_sel) and n_valid_wt1 > 1:
                    valid_wt1_sel_idx = np.where(valid_wt1_sel)[0]
                    valid_wt1_sel_arr = valid_wt1_arr[valid_wt1_sel_idx]
                    n_final_wt1 = len(valid_wt1_sel_arr)
                    
                    if n_final_wt1 > 1:
                        dots_wt1_wt1 = valid_wt1_sel_arr @ valid_wt1_sel_arr.T
                        wt1_compat = (dots_wt1_wt1 <= -1).astype(np.uint8)
                        np.fill_diagonal(wt1_compat, 1)
                        
                        # Greedy MIS
                        remaining = np.ones(n_final_wt1, dtype=np.uint8)
                        selected_wt1 = []
                        while remaining.any():
                            deg = (1 - wt1_compat) * remaining[np.newaxis, :]
                            deg = deg.sum(axis=1)
                            deg[remaining == 0] = n_final_wt1 + 1
                            min_deg = deg.min()
                            if min_deg >= n_final_wt1:
                                break
                            candidates = np.where((deg == min_deg) & (remaining > 0))[0]
                            if len(candidates) == 0:
                                break
                            pick = candidates[0]
                            selected_wt1.append(pick)
                            remaining[pick] = 0
                            remaining[1 - wt1_compat[pick]] = 0
                        
                        for idx in selected_wt1:
                            vectors.append(wt1_candidates[valid_wt1_idx[valid_wt1_sel_idx[idx]]])
    
    # Convert to array
    if len(vectors) == 0:
        # Fallback: return at least one vector
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    result = np.array(vectors, dtype=np.int64)
    return result
