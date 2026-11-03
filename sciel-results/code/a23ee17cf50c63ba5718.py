import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    D = 11
    
    # 1. Generate Candidate Pool
    # Weight-4 vectors (norm^2 = 4).
    # Constraint: ||v_i - v_j||^2 >= 4 => dot(v_i, v_j) <= 2.
    
    candidates_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
    
    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)
    
    # 2. Precompute Conflict Matrix
    # conflict[i, j] = 1 if dot(candidates[i], candidates[j]) > 2
    c_int = candidates.astype(np.int32)
    dot = c_int @ c_int.T
    
    # Conflict if dot > 2
    conflict = (dot > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    # Compatibility matrix (1 if compatible)
    compat = 1 - conflict
    np.fill_diagonal(compat, 1)
    
    # Precompute degrees
    degrees = conflict.sum(axis=1)
    
    best_count = 0
    best_selected = np.zeros(n, dtype=np.uint8)
    
    # 3. Multi-Start Search
    # Strategy:
    # A) Randomized Greedy Construction (Min-Degree heuristic)
    # B) Local Search Refinement (1-for-2, 1-for-3 moves)
    
    num_restarts = 500
    rng = np.random.default_rng(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # --- Phase 1: Randomized Greedy Construction ---
        selected = np.zeros(n, dtype=np.uint8)
        remaining = np.ones(n, dtype=np.uint8)
        
        count = 0
        while True:
            if time.time() > deadline - 5.0:
                break
                
            rem_idx = np.where(remaining > 0)[0]
            if len(rem_idx) == 0:
                break
                
            # Calculate effective degree in the remaining subgraph
            # deg[i] = sum(conflict[i, j] for j in remaining)
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n + 1 # Mask out non-remaining
            
            min_deg = deg.min()
            if min_deg >= n:
                break
                
            # Pick randomly from those with min degree
            candidates_to_pick = np.where((deg == min_deg) & (remaining > 0))[0]
            if len(candidates_to_pick) == 0:
                break
                
            pick_idx = rng.choice(candidates_to_pick)
            
            selected[pick_idx] = 1
            remaining[pick_idx] = 0
            # Update remaining: remove neighbors of pick_idx
            remaining[conflict[pick_idx] == 1] = 0
            count += 1
            
        # --- Phase 2: Local Search Refinement ---
        sel_indices = np.where(selected == 1)[0]
        if len(sel_indices) == 0:
            continue
            
        improved = True
        max_moves = 300
        move_count = 0
        
        while improved and move_count < max_moves and time.time() < deadline - 5.0:
            improved = False
            move_count += 1
            
            sel_indices = np.where(selected == 1)[0]
            k_sel = len(sel_indices)
            
            if k_sel == 0:
                break
            
            # Compute Total_Incompatibility for all nodes against current S
            # conflict[:, sel_indices] gives incompatibility matrix for S
            conflict_with_S = conflict[:, sel_indices]
            total_inc = conflict_with_S.sum(axis=1)
            
            # base_mask: nodes compatible with ALL S
            base_mask = (total_inc == 0) & (selected == 0)
            
            # For each v in S, determine valid candidates Cand_v
            # Cand_v = { x : x compatible with S\{v} }
            # x is in Cand_v if:
            # (total_inc[x] == 0 and not selected[x]) OR
            # (total_inc[x] == 1 and conflict_with_S[x, v_idx] == 1 and not selected[x])
            
            for v_idx_in_S, v in enumerate(sel_indices):
                if time.time() > deadline - 5.0:
                    break
                    
                # Construct Cand_v
                cond_mask = (total_inc == 1) & (conflict_with_S[:, v_idx_in_S] == 1) & (selected == 0)
                Cand_v = base_mask | cond_mask
                
                cand_indices = np.where(Cand_v > 0)[0]
                
                if len(cand_indices) == 0:
                    continue
                
                # Try 1-for-2: Find u, w in Cand_v such that compat[u, w] == 1
                if len(cand_indices) >= 2:
                    sub_conflict = conflict[np.ix_(cand_indices, cand_indices)]
                    sub_conflict_check = sub_conflict.copy()
                    np.fill_diagonal(sub_conflict_check, 1)
                    
                    if (sub_conflict_check == 0).any():
                        zero_rows, zero_cols = np.where(sub_conflict_check == 0)
                        if len(zero_rows) > 0:
                            u_local = zero_rows[0]
                            w_local = zero_cols[0]
                            u = cand_indices[u_local]
                            w = cand_indices[w_local]
                            
                            # Perform move: remove v, add u, w
                            selected[v] = 0
                            selected[u] = 1
                            selected[w] = 1
                            improved = True
                            break # Restart while loop
                
                # Try 1-for-1: Find u in Cand_v
                if not improved and len(cand_indices) > 0:
                    u = cand_indices[rng.integers(len(cand_indices))]
                    selected[v] = 0
                    selected[u] = 1
                    improved = True
                    break # Restart while loop
        
        # Final count
        current_count = selected.sum()
        if current_count > best_count:
            best_count = current_count
            best_selected = selected.copy()
            
    # 4. Extract Result
    sel_indices = np.where(best_selected == 1)[0]
    if len(sel_indices) == 0:
        # Fallback
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)
        
    result = candidates[sel_indices.astype(np.int64)]
    return result.astype(np.int64)

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} vectors")
    # Verify
    if len(res) > 1:
        norms = np.sum(res**2, axis=1)
        max_norm = np.max(norms)
        dots = res @ res.T
        np.fill_diagonal(dots, 0)
        # dist^2 = ni + nj - 2*dot
        # We need dist^2 >= max_norm
        # ni + nj - 2*dot >= max_norm
        # Since ni=nj=4 for all, 8 - 2*dot >= 4 => 2*dot <= 4 => dot <= 2
        min_dist_sq = np.inf
        for i in range(len(res)):
            for j in range(i+1, len(res)):
                d2 = norms[i] + norms[j] - 2*dots[i,j]
                if d2 < min_dist_sq:
                    min_dist_sq = d2
        print(f"Max norm: {max_norm}, Min dist^2: {min_dist_sq}")
        if min_dist_sq >= max_norm:
            print("Valid")
        else:
            print("INVALID")
