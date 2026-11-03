"""Support-level approach with tabu search and direct vector expansion."""
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118.0
    D = 11
    
    # Generate all C(11,4) = 330 supports
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)  # 330
    
    # Build conflict matrix: edge if intersection size == 3
    support_sets = [set(s) for s in supports]
    adj_mask = np.zeros((n_sup, n_sup), dtype=np.uint8)
    for i in range(n_sup):
        for j in range(i + 1, n_sup):
            if len(support_sets[i] & support_sets[j]) == 3:
                adj_mask[i, j] = 1
                adj_mask[j, i] = 1
    
    # Tabu search on support-level MIS
    def tabu_search(init_S, n_iters=20000, tabu_len=30, seed=0):
        rng = np.random.default_rng(seed)
        rand = random.Random(seed)
        
        S = init_S.copy()
        current_size = int(S.sum())
        best_size = current_size
        best_S = S.copy()
        
        # Tabu list: (index, iteration_when_became_free)
        tabu = {}
        iter_count = 0
        
        for it in range(n_iters):
            if time.time() > deadline - 20.0:
                break
            iter_count += 1
            
            # Clean expired tabu entries
            for idx in list(tabu.keys()):
                if iter_count >= tabu[idx]:
                    del tabu[idx]
            
            # Try all possible moves: add one support, or swap one in for one out
            best_move = None
            best_delta = 0
            
            # Add moves
            for v in range(n_sup):
                if S[v]:
                    continue
                if v in tabu:
                    continue
                conflicts = np.where(adj_mask[v] & S)[0]
                delta = 1 - len(conflicts)
                if delta > best_delta:
                    best_delta = delta
                    best_move = ('add', v, conflicts)
            
            # Swap moves: remove one selected, add one unselected
            sel_idx = np.where(S == 1)[0]
            if len(sel_idx) > 0:
                # Sample some swaps to keep it fast
                n_swaps = min(200, len(sel_idx) * 10)
                for _ in range(n_swaps):
                    rem_v = rng.choice(sel_idx)
                    if rem_v in tabu:
                        continue
                    add_v = rng.integers(0, n_sup)
                    if S[add_v] or add_v == rem_v:
                        continue
                    if add_v in tabu:
                        continue
                    
                    # Check if add_v conflicts with (S - rem_v)
                    S_temp = S.copy()
                    S_temp[rem_v] = 0
                    conflicts = np.where(adj_mask[add_v] & S_temp)[0]
                    delta = 1 - len(conflicts)  # remove 1, add 1, minus conflicts
                    if delta > best_delta:
                        best_delta = delta
                        best_move = ('swap', rem_v, add_v, conflicts)
            
            if best_move is None:
                # Random move
                if len(sel_idx) > 0:
                    rem_v = rng.choice(sel_idx)
                    add_v = rng.integers(0, n_sup)
                    while S[add_v] or add_v == rem_v:
                        add_v = rng.integers(0, n_sup)
                    S_temp = S.copy()
                    S_temp[rem_v] = 0
                    conflicts = np.where(adj_mask[add_v] & S_temp)[0]
                    delta = 1 - len(conflicts)
                    best_move = ('swap', rem_v, add_v, conflicts)
                    best_delta = delta
                else:
                    break
            
            # Execute move
            if best_move[0] == 'add':
                _, v, conflicts = best_move
                S[conflicts] = 0
                S[v] = 1
                tabu[v] = iter_count + tabu_len
                for c in conflicts:
                    tabu[c] = iter_count + tabu_len
            else:
                _, rem_v, add_v, conflicts = best_move
                S[rem_v] = 0
                S[conflicts] = 0
                S[add_v] = 1
                tabu[rem_v] = iter_count + tabu_len
                tabu[add_v] = iter_count + tabu_len
                for c in conflicts:
                    tabu[c] = iter_count + tabu_len
            
            current_size = int(S.sum())
            if current_size > best_size:
                best_size = current_size
                best_S = S.copy()
        
        return best_S, best_size
    
    # Greedy initialization
    def greedy_is(seed=0):
        rng = np.random.default_rng(seed)
        remaining = np.ones(n_sup, dtype=np.uint8)
        selected = np.zeros(n_sup, dtype=np.uint8)
        
        while True:
            if time.time() > deadline - 25.0:
                break
            rem_idx = np.where(remaining > 0)[0]
            if len(rem_idx) == 0:
                break
            
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
    
    # Main search
    best_S = np.zeros(n_sup, dtype=np.uint8)
    best_size = 0
    
    n_restarts = 40
    for i in range(n_restarts):
        if time.time() > deadline - 35.0:
            break
        
        S_init = greedy_is(seed=1000 + i)
        init_size = int(S_init.sum())
        
        # Tabu search
        S_tabu, tabu_size = tabu_search(S_init, n_iters=50000, tabu_len=40, seed=2000 + i)
        
        if tabu_size > best_size:
            best_size = tabu_size
            best_S = S_tabu.copy()
        
        # Also try with different tabu lengths
        if i % 3 == 0 and time.time() < deadline - 30.0:
            S_tabu2, tabu_size2 = tabu_search(S_init, n_iters=80000, tabu_len=20, seed=5000 + i)
            if tabu_size2 > best_size:
                best_size = tabu_size2
                best_S = S_tabu2.copy()
    
    # Expand to vectors
    selected_supports = np.where(best_S == 1)[0]
    
    vectors = []
    for sup_idx in selected_supports:
        positions = supports[sup_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            vectors.append(v)
    
    # Try to add weight-2 and weight-1 vectors
    if len(vectors) > 0:
        wt4_arr = np.array(vectors, dtype=np.int32)
        
        # Weight-2 candidates
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
        
        # Weight-1 candidates
        wt1_candidates = []
        for pos in range(D):
            for s in [1, -1]:
                v = [0] * D
                v[pos] = s
                wt1_candidates.append(v)
        
        if len(wt1_candidates) > 0:
            wt1_arr = np.array(wt1_candidates, dtype=np.int32)
            dots_wt1 = wt1_arr @ wt4_arr.T
            valid_wt1 = np.all(dots_wt1 <= 0, axis=1)
            
            if np.any(valid_wt1):
                valid_wt1_idx = np.where(valid_wt1)[0]
                valid_wt1_arr = wt1_arr[valid_wt1_idx]
                n_valid_wt1 = len(valid_wt1_arr)
                
                selected_arr = np.array(vectors, dtype=np.int32)
                n_sel = len(selected_arr)
                
                if n_sel > 0:
                    dots_wt1_sel = valid_wt1_arr @ selected_arr.T
                    sel_norms_sq = np.sum(selected_arr ** 2, axis=1)
                    thresholds = (sel_norms_sq - 3) / 2.0
                    valid_wt1_sel = np.all(dots_wt1_sel <= thresholds[np.newaxis, :], axis=1)
                else:
                    valid_wt1_sel = np.ones(n_valid_wt1, dtype=bool)
                
                if np.any(valid_wt1_sel) and n_valid_wt1 > 1:
                    valid_wt1_sel_idx = np.where(valid_wt1_sel)[0]
                    valid_wt1_sel_arr = valid_wt1_arr[valid_wt1_sel_idx]
                    n_final_wt1 = len(valid_wt1_sel_arr)
                    
                    if n_final_wt1 > 1:
                        dots_wt1_wt1 = valid_wt1_sel_arr @ valid_wt1_sel_arr.T
                        wt1_compat = (dots_wt1_wt1 <= -1).astype(np.uint8)
                        np.fill_diagonal(wt1_compat, 1)
                        
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
    
    if len(vectors) == 0:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    result = np.array(vectors, dtype=np.int64)
    return result
