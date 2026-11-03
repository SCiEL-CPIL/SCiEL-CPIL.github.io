import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s kill

    # 1. Generate Candidate Pool: Weight-4 vectors (Type B)
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
    conflict = (dot > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    # Convert to list of lists of neighbors for faster iteration?
    # Or stick to numpy arrays.
    
    rng = np.random.default_rng(42)
    
    best_count = 0
    best_selected = np.zeros(n, dtype=np.uint8)
    
    def grasp_construct(alpha):
        selected = np.zeros(n, dtype=np.uint8)
        remaining = np.ones(n, dtype=np.uint8)
        
        count = 0
        while True:
            if time.time() > deadline - 10.0:
                break
                
            rem_idx = np.where(remaining > 0)[0]
            if len(rem_idx) == 0:
                break
                
            # Calculate degrees in remaining subgraph
            # deg[i] = number of neighbors in remaining
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n + 1
            
            min_deg = deg.min()
            max_deg = deg[remaining > 0].max()
            
            if min_deg > n:
                break
                
            # RCL: vertices with degree <= min_deg + alpha * (max_deg - min_deg)
            threshold = min_deg + alpha * (max_deg - min_deg)
            rcl_candidates = np.where((deg <= threshold) & (remaining > 0))[0]
            
            if len(rcl_candidates) == 0:
                break
                
            pick_idx = rng.choice(rcl_candidates)
            
            selected[pick_idx] = 1
            remaining[pick_idx] = 0
            # Remove neighbors
            remaining[conflict[pick_idx] == 1] = 0
            count += 1
            
        return selected, count

    def local_search(selected):
        sel_indices = np.where(selected == 1)[0]
        if len(sel_indices) == 0:
            return selected, 0
            
        improved = True
        max_moves = 50
        move_count = 0
        current_count = len(sel_indices)
        
        while improved and move_count < max_moves and time.time() < deadline - 5.0:
            improved = False
            move_count += 1
            
            sel_indices = np.where(selected == 1)[0]
            k_sel = len(sel_indices)
            
            if k_sel == 0:
                break
            
            # Compute conflict with S for all nodes
            conflict_with_S = conflict[:, sel_indices]
            total_inc = conflict_with_S.sum(axis=1)
            
            # base_mask: nodes compatible with ALL S
            base_mask = (total_inc == 0) & (selected == 0)
            
            # Try 1-for-2 moves (Add 2, Remove 1)
            # This increases count by 1.
            order = rng.permutation(k_sel)
            
            for idx in order:
                if time.time() > deadline - 5.0:
                    break
                    
                v = sel_indices[idx]
                
                # Nodes compatible with S \ {v}
                # If total_inc[x] == 0, it's compatible with S, so compatible with S\{v}
                # If total_inc[x] == 1 and conflict_with_S[x, idx] == 1, it's only incompatible with v
                cond_mask = (total_inc == 1) & (conflict_with_S[:, idx] == 1) & (selected == 0)
                Cand_v = base_mask | cond_mask
                
                cand_indices = np.where(Cand_v > 0)[0]
                
                if len(cand_indices) < 2:
                    continue
                
                # Check if there is any compatible pair in cand_indices
                # We want u, w in cand_indices such that conflict[u, w] == 0
                sub_conflict = conflict[np.ix_(cand_indices, cand_indices)]
                # We need a pair (u, w) with u != w and sub_conflict[u, w] == 0
                # Since diagonal is 0 in sub_conflict (if we don't set it), we need to be careful.
                # conflict[i,i] is 0. So we need off-diagonal 0.
                
                # Find any zero in sub_conflict where i != j
                # Since it's symmetric, we can just look for any 0.
                # But we must ensure i != j.
                # np.where(sub_conflict == 0) returns (i, j).
                # If i == j, it's the diagonal.
                
                zero_rows, zero_cols = np.where(sub_conflict == 0)
                found_pair = False
                for r, c in zip(zero_rows, zero_cols):
                    if r != c:
                        u = cand_indices[r]
                        w = cand_indices[c]
                        
                        # Perform move
                        selected[v] = 0
                        selected[u] = 1
                        selected[w] = 1
                        improved = True
                        current_count += 1
                        found_pair = True
                        break
                        
                if found_pair:
                    break # Restart local search loop
            
            # If no 1-for-2 found, try 1-for-1 swaps to escape local optima
            if not improved:
                # Random 1-for-1 swaps
                for _ in range(5):
                    if time.time() > deadline - 5.0:
                        break
                    v_idx = rng.choice(k_sel)
                    v = sel_indices[v_idx]
                    
                    cond_mask = (total_inc == 1) & (conflict_with_S[:, v_idx] == 1) & (selected == 0)
                    base_mask = (total_inc == 0) & (selected == 0)
                    Cand_v = base_mask | cond_mask
                    cand_indices = np.where(Cand_v > 0)[0]
                    
                    if len(cand_indices) > 0:
                        u = rng.choice(cand_indices)
                        # Swap
                        selected[v] = 0
                        selected[u] = 1
                        # current_count unchanged
                        # This is a neutral move. We break to re-evaluate 1-for-2
                        break

        return selected, current_count

    # Run GRASP iterations
    alphas = [0.1, 0.2, 0.5, 0.8, 1.0]
    
    while time.time() < deadline - 20.0:
        alpha = rng.choice(alphas)
        sel, cnt = grasp_construct(alpha)
        if cnt > best_count:
            best_count = cnt
            best_selected = sel.copy()
            
        # Refine
        if cnt > 0:
            sel_refined, cnt_refined = local_search(sel)
            if cnt_refined > best_count:
                best_count = cnt_refined
                best_selected = sel_refined.copy()

    # 5. Add Type A vectors
    type_a_vectors = []
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            type_a_vectors.append(v)
            
    type_a_arr = np.array(type_a_vectors, dtype=np.int64)
    
    # 6. Combine and Verify
    if best_count > 0:
        type_b_selected = candidates[best_selected == 1].astype(np.int64)
        all_vectors = np.vstack([type_a_arr, type_b_selected])
    else:
        all_vectors = type_a_arr
        
    # Verification
    norms_sq = np.sum(all_vectors * all_vectors, axis=1)
    R_sq = np.max(norms_sq)
    
    k = len(all_vectors)
    dots = all_vectors @ all_vectors.T
    num = norms_sq[:, None] + norms_sq[None, :] - R_sq
    thresh = num // 2
    
    valid = True
    for i in range(k):
        for j in range(i+1, k):
            if dots[i, j] > thresh[i, j]:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        return type_a_arr
        
    return all_vectors
