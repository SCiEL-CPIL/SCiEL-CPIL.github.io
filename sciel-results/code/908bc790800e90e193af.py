import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Generate weight-4 vectors: four ±1's, rest 0 → C(11,4)*2^4 = 5280
    candidates_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
    
    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)
    
    # Precompute conflict matrix: conflict[i,j]=1 if dot(c_i, c_j) > 2
    c_int = candidates.astype(np.int32)
    dot = c_int @ c_int.T
    conflict = (dot > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    best_count = 0
    best_selected = np.zeros(n, dtype=np.uint8)
    
    rng = np.random.default_rng(42)
    
    def greedy_construct(alpha=0.3):
        """GRASP-style construction with DENSITY-BASED (max-degree) ordering.
        Prioritizes high-degree nodes to aggressively block competitors early,
        creating larger gaps for subsequent selections."""
        selected = np.zeros(n, dtype=np.uint8)
        remaining = np.ones(n, dtype=np.uint8)
        count = 0
        
        while True:
            if time.time() > deadline - 10.0:
                break
            rem_idx = np.where(remaining > 0)[0]
            if len(rem_idx) == 0:
                break
            
            # Compute degree (number of conflicts) in remaining subgraph
            deg = (conflict[rem_idx] @ remaining).astype(np.int32)
            min_deg = deg.min()
            max_deg = deg.max()
            
            # DENSITY-BASED: prioritize HIGH degree nodes to block competitors early
            # Restricted candidate list: nodes with degree in [max_deg - alpha*(max_deg-min_deg), max_deg]
            if max_deg == min_deg:
                rcl = rem_idx
            else:
                threshold = max_deg - alpha * (max_deg - min_deg)
                rcl = rem_idx[deg >= threshold]
                if len(rcl) == 0:
                    rcl = rem_idx[deg == max_deg]
            
            pick_idx = rng.choice(rcl)
            
            selected[pick_idx] = 1
            remaining[pick_idx] = 0
            remaining[conflict[pick_idx] == 1] = 0
            count += 1
        
        return selected, count
    
    def local_search(selected):
        """1-for-2 and 1-for-1 swap local search."""
        sel_indices = np.where(selected == 1)[0]
        if len(sel_indices) == 0:
            return selected, 0
        
        improved = True
        max_moves = 300
        move_count = 0
        moves_made = 0
        no_improve_streak = 0
        
        while improved and move_count < max_moves and time.time() < deadline - 5.0:
            improved = False
            move_count += 1
            
            sel_indices = np.where(selected == 1)[0]
            k_sel = len(sel_indices)
            if k_sel == 0:
                break
            
            # Compute incompatibility with current set S
            conflict_with_S = conflict[:, sel_indices]
            total_inc = conflict_with_S.sum(axis=1)
            
            # Base mask: nodes compatible with ALL of S
            base_mask = (total_inc == 0) & (selected == 0)
            
            # Try 1-for-2 moves: remove v from S, add u,w compatible with S\{v} and each other
            order = rng.permutation(k_sel)
            found_move = False
            
            for idx in order:
                if time.time() > deadline - 5.0:
                    break
                    
                v = sel_indices[idx]
                
                # Cand_v: nodes compatible with S \ {v}
                # x is compatible with S\{v} if:
                #   total_inc[x] == 0 (compatible with all S), OR
                #   total_inc[x] == 1 and the only conflict is with v
                cond_mask = (total_inc == 1) & (conflict_with_S[:, idx] == 1) & (selected == 0)
                Cand_v = base_mask | cond_mask
                
                cand_indices = np.where(Cand_v > 0)[0]
                if len(cand_indices) < 2:
                    continue
                
                # Find a compatible pair in cand_indices
                sub_conflict = conflict[np.ix_(cand_indices, cand_indices)]
                sub_conflict_check = sub_conflict.copy()
                np.fill_diagonal(sub_conflict_check, 1)
                
                zero_positions = np.where(sub_conflict_check == 0)
                if len(zero_positions[0]) > 0:
                    u_local = zero_positions[0][0]
                    w_local = zero_positions[1][0]
                    u = cand_indices[u_local]
                    w = cand_indices[w_local]
                    
                    # Verify u != w
                    if u != w:
                        selected[v] = 0
                        selected[u] = 1
                        selected[w] = 1
                        improved = True
                        moves_made += 1
                        no_improve_streak = 0
                        found_move = True
                        break
            
            # If no 1-for-2 found, try 1-for-1 swaps to escape local optima
            if not found_move and k_sel > 0:
                # Try a few random 1-for-1 swaps
                for _ in range(20):
                    if time.time() > deadline - 5.0:
                        break
                    
                    v_idx = rng.integers(k_sel)
                    v = sel_indices[v_idx]
                    
                    # Find candidates compatible with S \ {v}
                    cond_mask = (total_inc == 1) & (conflict_with_S[:, v_idx] == 1) & (selected == 0)
                    base_mask2 = (total_inc == 0) & (selected == 0)
                    Cand_v = base_mask2 | cond_mask
                    cand_indices = np.where(Cand_v > 0)[0]
                    
                    if len(cand_indices) == 0:
                        continue
                    
                    u = cand_indices[rng.integers(len(cand_indices))]
                    
                    # Perform swap: remove v, add u
                    selected[v] = 0
                    selected[u] = 1
                    improved = True
                    moves_made += 1
                    no_improve_streak = 0
                    break
                else:
                    no_improve_streak += 1
                    if no_improve_streak > 5:
                        break
        
        final_count = selected.sum()
        return selected, int(final_count)
    
    # Multi-start GRASP with density-based ordering
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Vary alpha for diversity: smaller alpha = more aggressive max-degree
        alpha = 0.1 + 0.6 * (r % 10) / 10.0
        
        sel, cnt = greedy_construct(alpha=alpha)
        sel, cnt = local_search(sel)
        
        if cnt > best_count:
            best_count = cnt
            best_selected = sel.copy()
        
        # Early termination if we've found a very good solution
        if best_count > 1400 and r > 10:
            break
    
    if best_count == 0:
        # Fallback: simple greedy
        sel, cnt = greedy_construct(alpha=0.0)
        best_selected = sel
        best_count = cnt
    
    result = candidates[best_selected == 1].astype(np.int64)
    
    # Verify validity
    if len(result) > 1:
        n_res = len(result)
        max_norm_sq = int(np.max(np.sum(result**2, axis=1)))
        valid = True
        # Vectorized check
        diffs = result[:, np.newaxis, :] - result[np.newaxis, :, :]
        dist_sq = np.sum(diffs**2, axis=2)
        np.fill_diagonal(dist_sq, max_norm_sq)
        if np.min(dist_sq) < max_norm_sq:
            valid = False
        
        if not valid:
            # Fallback to a known valid configuration
            vecs = []
            for i in range(D):
                for s in [1, -1]:
                    v = [0] * D
                    v[i] = 2 * s
                    vecs.append(v)
            return np.array(vecs, dtype=np.int64)
    
    return result
