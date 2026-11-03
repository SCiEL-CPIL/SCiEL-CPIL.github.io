"""Combined: support-level MIS + enhanced SA with 2-for-2 moves + vector packing for kissing number in dimension 11."""
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # safety margin before 150s limit
    
    D = 11
    
    # Generate all C(11,4) = 330 supports
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)
    
    # Build conflict matrix: edge if intersection size == 3
    support_sets = [set(s) for s in supports]
    adj_mask = np.zeros((n_sup, n_sup), dtype=np.uint8)
    for i in range(n_sup):
        for j in range(i + 1, n_sup):
            if len(support_sets[i] & support_sets[j]) == 3:
                adj_mask[i, j] = 1
                adj_mask[j, i] = 1
    
    # Precompute adjacency lists as arrays for faster access
    adj_lists = [np.where(adj_mask[i] == 1)[0] for i in range(n_sup)]
    
    def greedy_is(seed=0):
        """Greedy independent set using minimum-degree heuristic with random tie-breaking."""
        rng = np.random.default_rng(seed)
        remaining = np.ones(n_sup, dtype=np.uint8)
        selected = np.zeros(n_sup, dtype=np.uint8)
        
        while True:
            if time.time() > deadline - 12.0:
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
    
    def local_search_sa(selected, n_steps=50000, seed=0, use_2for2=True):
        """Simulated annealing with 1-for-k and 2-for-2 moves."""
        rng = np.random.default_rng(seed)
        rand = random.Random(seed)
        
        S = selected.copy()
        current_size = int(S.sum())
        best_size = current_size
        best_S = S.copy()
        
        T = 1.5
        cooling = 0.9998
        min_T = 0.03
        
        step = 0
        while step < n_steps and time.time() < deadline - 8.0:
            step += 1
            
            # Decide move type: 1-for-k (70%) or 2-for-2 (30%)
            if use_2for2 and rand.random() < 0.3:
                # 2-for-2 move: remove 2 random selected, add 2 random unselected
                sel_idx = np.where(S == 1)[0]
                if len(sel_idx) < 2:
                    continue
                i1, i2 = rng.integers(0, len(sel_idx), size=2)
                if i1 == i2:
                    continue
                v1, v2 = sel_idx[i1], sel_idx[i2]
                
                # Find candidates: not in S, not conflicting with v1 or v2, and compatible with each other
                # and compatible with all other selected (except v1, v2)
                others = np.delete(sel_idx, [i1, i2])
                
                # Candidates must not conflict with v1, v2, or any other selected
                compat = (adj_mask[v1] == 0) & (adj_mask[v2] == 0) & (S == 0)
                if len(others) > 0:
                    for o in others:
                        compat &= (adj_mask[o] == 0)
                
                candidates = np.where(compat)[0]
                if len(candidates) < 2:
                    continue
                
                # Try to find a pair (u, w) from candidates that are mutually compatible
                found = False
                for trial in range(20):
                    u_idx = rng.integers(0, len(candidates))
                    w_idx = rng.integers(0, len(candidates) - 1)
                    if w_idx >= u_idx:
                        w_idx += 1
                    u, w = candidates[u_idx], candidates[w_idx]
                    if adj_mask[u, w] == 0:
                        # Check compatibility with others
                        if len(others) > 0:
                            if np.any(adj_mask[u][others] == 1) or np.any(adj_mask[w][others] == 1):
                                continue
                        S[v1] = 0
                        S[v2] = 0
                        S[u] = 1
                        S[w] = 1
                        # Size unchanged (2 removed, 2 added)
                        found = True
                        break
                
                if not found:
                    # Try a different approach: just do 1-for-k
                    v = rng.integers(0, n_sup)
                    if S[v]:
                        continue
                    conflicts = adj_mask[v] & S
                    n_conf = int(conflicts.sum())
                    if n_conf == 0:
                        S[v] = 1
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_S = S.copy()
                    elif n_conf > 0:
                        delta = 1 - n_conf
                        if delta >= 0:
                            S[conflicts] = 0
                            S[v] = 1
                            current_size += delta
                            if current_size > best_size:
                                best_size = current_size
                                best_S = S.copy()
                        elif T > min_T:
                            p = np.exp(delta / T)
                            if rand.random() < p:
                                S[conflicts] = 0
                                S[v] = 1
                                current_size += delta
                                if current_size > best_size:
                                    best_size = current_size
                                    best_S = S.copy()
            else:
                # 1-for-k move
                v = rng.integers(0, n_sup)
                
                if S[v]:
                    # Occasionally remove a vertex to allow reconfiguration
                    if rand.random() < 0.02:
                        S[v] = 0
                        current_size -= 1
                    continue
                
                conflicts = adj_mask[v] & S
                n_conf = int(conflicts.sum())
                
                if n_conf == 0:
                    S[v] = 1
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S.copy()
                else:
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
    
    def polish(S, max_rounds=30):
        """Iterative 2-for-2 polish: try to find improving swaps."""
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 5.0:
            improved = False
            rounds += 1
            
            sel = np.where(S == 1)[0]
            k = len(sel)
            if k < 2:
                break
            
            # Try multiple random pairs
            n_trials = min(2000, k * k // 2)
            for _ in range(n_trials):
                if time.time() > deadline - 5.0:
                    break
                i1, i2 = np.random.default_rng().integers(0, k, size=2)
                if i1 == i2:
                    continue
                v1, v2 = sel[i1], sel[i2]
                
                others = np.delete(sel, [i1, i2])
                
                # Find candidates compatible with v1, v2, and all others
                compat = (adj_mask[v1] == 0) & (adj_mask[v2] == 0) & (S == 0)
                if len(others) > 0:
                    for o in others:
                        compat &= (adj_mask[o] == 0)
                
                candidates = np.where(compat)[0]
                if len(candidates) < 2:
                    continue
                
                # Try to find a mutually compatible pair
                for trial in range(30):
                    u_idx = np.random.choice(len(candidates))
                    w_idx = np.random.choice(len(candidates) - 1)
                    if w_idx >= u_idx:
                        w_idx += 1
                    u, w = candidates[u_idx], candidates[w_idx]
                    if adj_mask[u, w] == 0:
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
    
    # Main search: multiple restarts with varying SA parameters
    best_S = np.zeros(n_sup, dtype=np.uint8)
    best_size = 0
    
    n_restarts = 40
    for i in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Greedy initialization with different seeds
        S_init = greedy_is(seed=1000 + i * 7)
        init_size = int(S_init.sum())
        
        # SA with different step counts and cooling rates
        if i < 10:
            n_steps = 40000
            seed_sa = 2000 + i
        elif i < 25:
            n_steps = 60000
            seed_sa = 5000 + i
        else:
            n_steps = 80000
            seed_sa = 8000 + i
        
        S_sa, sa_size = local_search_sa(S_init, n_steps=n_steps, seed=seed_sa, use_2for2=True)
        
        # Polish
        S_polished = polish(S_sa.copy(), max_rounds=15)
        pol_size = int(S_polished.sum())
        
        if pol_size > best_size:
            best_size = pol_size
            best_S = S_polished.copy()
        
        # Every 5th restart, try a longer SA from the best found so far
        if i % 5 == 4 and time.time() < deadline - 25.0:
            S_long, long_size = local_search_sa(best_S.copy(), n_steps=100000, seed=10000 + i, use_2for2=True)
            S_long_pol = polish(S_long.copy(), max_rounds=20)
            long_pol_size = int(S_long_pol.sum())
            if long_pol_size > best_size:
                best_size = long_pol_size
                best_S = S_long_pol.copy()
    
    # Expand support-level solution to vectors
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
    
    # Convert to array for dot product checks
    if len(vectors) > 0:
        wt4_arr = np.array(vectors, dtype=np.int32)
        n_wt4 = len(wt4_arr)
        
        # Try to add weight-2 vectors
        wt2_candidates = []
        for positions in itertools.combinations(range(D), 2):
            for signs in itertools.product([1, -1], repeat=2):
                v = [0] * D
                for pos, s in zip(positions, signs):
                    v[pos] = s
                wt2_candidates.append(v)
        
        if len(wt2_candidates) > 0:
            wt2_arr = np.array(wt2_candidates, dtype=np.int32)
            dots_wt2 = wt2_arr @ wt4_arr.T
            valid_wt2 = np.all(dots_wt2 <= 1, axis=1)
            
            if np.any(valid_wt2):
                valid_wt2_idx = np.where(valid_wt2)[0]
                valid_wt2_arr = wt2_arr[valid_wt2_idx]
                n_valid_wt2 = len(valid_wt2_arr)
                
                if n_valid_wt2 > 1:
                    dots_wt2_wt2 = valid_wt2_arr @ valid_wt2_arr.T
                    wt2_compat = (dots_wt2_wt2 <= 0).astype(np.uint8)
                    np.fill_diagonal(wt2_compat, 1)
                    
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
        
        # Try to add weight-1 vectors
        wt1_candidates = []
        for pos in range(D):
            for s in [1, -1]:
                v = [0] * D
                v[pos] = s
                wt1_candidates.append(v)
        
        if len(wt1_candidates) > 0:
            wt1_arr = np.array(wt1_candidates, dtype=np.int32)
            
            # Check against all current vectors
            current_arr = np.array(vectors, dtype=np.int32)
            n_cur = len(current_arr)
            cur_norms_sq = np.sum(current_arr ** 2, axis=1)
            
            dots_wt1 = wt1_arr @ current_arr.T
            # Condition: v.s <= (||v||^2 + ||s||^2 - 4) / 2 = (1 + cur_norms_sq - 4)/2 = (cur_norms_sq - 3)/2
            thresholds = (cur_norms_sq - 3) / 2.0
            valid_wt1 = np.all(dots_wt1 <= thresholds[np.newaxis, :], axis=1)
            
            if np.any(valid_wt1):
                valid_wt1_idx = np.where(valid_wt1)[0]
                valid_wt1_arr = wt1_arr[valid_wt1_idx]
                n_valid_wt1 = len(valid_wt1_arr)
                
                if n_valid_wt1 > 1:
                    dots_wt1_wt1 = valid_wt1_arr @ valid_wt1_arr.T
                    wt1_compat = (dots_wt1_wt1 <= -1).astype(np.uint8)
                    np.fill_diagonal(wt1_compat, 1)
                    
                    remaining = np.ones(n_valid_wt1, dtype=np.uint8)
                    selected_wt1 = []
                    while remaining.any():
                        deg = (1 - wt1_compat) * remaining[np.newaxis, :]
                        deg = deg.sum(axis=1)
                        deg[remaining == 0] = n_valid_wt1 + 1
                        min_deg = deg.min()
                        if min_deg >= n_valid_wt1:
                            break
                        candidates = np.where((deg == min_deg) & (remaining > 0))[0]
                        if len(candidates) == 0:
                            break
                        pick = candidates[0]
                        selected_wt1.append(pick)
                        remaining[pick] = 0
                        remaining[1 - wt1_compat[pick]] = 0
                    
                    for idx in selected_wt1:
                        vectors.append(wt1_candidates[valid_wt1_idx[idx]])
    
    if len(vectors) == 0:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    result = np.array(vectors, dtype=np.int64)
    return result
