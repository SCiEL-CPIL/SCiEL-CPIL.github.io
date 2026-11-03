import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s kill

    # 1. Generate Candidate Pool: Weight-4 vectors (Type B)
    # Norm^2 = 4.
    # Constraint for Type B vs Type B: dot <= 2.
    # We identified that Type A (weight 1, value 2) are compatible with ALL Type B and each other.
    # So we solve for MIS on Type B, then add all Type A.
    
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
    
    # 3. Greedy Construction with Min-Degree Heuristic
    # We will run this multiple times with random tie-breaking to find a good seed.
    
    rng = np.random.default_rng(42)
    
    best_count = 0
    best_selected = np.zeros(n, dtype=np.uint8)
    
    def greedy_construct():
        selected = np.zeros(n, dtype=np.uint8)
        remaining = np.ones(n, dtype=np.uint8)
        
        count = 0
        while True:
            if time.time() > deadline - 15.0:
                break
                
            rem_idx = np.where(remaining > 0)[0]
            if len(rem_idx) == 0:
                break
                
            # Calculate effective degree in the remaining subgraph
            # deg[i] = sum of conflicts with other remaining nodes
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            # Nodes not remaining have high degree to exclude them
            deg[remaining == 0] = n + 1
            
            min_deg = deg.min()
            if min_deg > n:
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
            
        return selected, count

    # Run greedy multiple times to find a good starting point
    for i in range(20):
        if time.time() > deadline - 20.0:
            break
        sel, cnt = greedy_construct()
        if cnt > best_count:
            best_count = cnt
            best_selected = sel.copy()

    # 4. Local Search Refinement
    # Try 1-for-2 swaps: Remove v from S, add u, w not in S.
    # Condition: u, w compatible with S\{v} and with each other.
    
    def local_search(selected):
        sel_indices = np.where(selected == 1)[0]
        if len(sel_indices) == 0:
            return selected, 0
            
        improved = True
        max_moves = 100
        move_count = 0
        moves_made = 0
        current_count = len(sel_indices)
        
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
            
            # Try 1-for-2 moves
            order = rng.permutation(k_sel)
            
            for idx in order:
                if time.time() > deadline - 5.0:
                    break
                    
                v = sel_indices[idx]
                
                # Construct Cand_v: nodes compatible with S \ {v}
                # x is in Cand_v if:
                # (total_inc[x] == 0 and not selected[x]) OR
                # (total_inc[x] == 1 and conflict_with_S[x, idx] == 1 and not selected[x])
                
                cond_mask = (total_inc == 1) & (conflict_with_S[:, idx] == 1) & (selected == 0)
                Cand_v = base_mask | cond_mask
                
                cand_indices = np.where(Cand_v > 0)[0]
                
                if len(cand_indices) < 2:
                    continue
                
                # Check if there is any compatible pair in cand_indices
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
                        moves_made += 1
                        current_count += 1
                        break # Restart while loop with new S
                        
            # If no 1-for-2 found, try 1-for-1 swaps to escape local optima
            if not improved:
                # Find a v in S and a u not in S such that u is compatible with S\{v}
                # We prioritize swaps that might open up future 1-for-2 opportunities
                # Heuristic: pick v with high degree in S, pick u with low degree in S\{v}
                
                # For simplicity and speed, we do a limited random search for 1-for-1
                # that results in a "better" neighborhood for the remaining nodes
                # Actually, just doing random 1-for-1 can cause cycles.
                # We will skip complex 1-for-1 unless we have time, as 1-for-2 is the primary driver.
                # But to escape dead ends, let's try a few random 1-for-1s.
                
                for _ in range(10):
                    if time.time() > deadline - 5.0:
                        break
                    v_idx = rng.choice(k_sel)
                    v = sel_indices[v_idx]
                    
                    # Find candidates compatible with S\{v}
                    cond_mask = (total_inc == 1) & (conflict_with_S[:, v_idx] == 1) & (selected == 0)
                    base_mask = (total_inc == 0) & (selected == 0)
                    Cand_v = base_mask | cond_mask
                    cand_indices = np.where(Cand_v > 0)[0]
                    
                    if len(cand_indices) > 0:
                        u = rng.choice(cand_indices)
                        # Only swap if it doesn't immediately reduce options?
                        # Hard to judge. Just do it.
                        selected[v] = 0
                        selected[u] = 1
                        # current_count unchanged
                        # We mark improved to restart loop, but this is a neutral move.
                        # To prevent infinite loops of neutral moves, we rely on the
                        # fact that the next 1-for-2 might now be possible.
                        # If no 1-for-2 is found in the next few iterations, we stop.
                        # We'll just break out of the 1-for-1 loop and let the main loop
                        # check for 1-for-2 again.
                        break

        return selected, current_count

    # Run local search on the best greedy solution
    if best_count > 0:
        best_selected, final_count_b = local_search(best_selected)
        
    # 5. Add Type A vectors
    # Type A: ±2e_i
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
    # Norms
    norms_sq = np.sum(all_vectors * all_vectors, axis=1)
    R_sq = np.max(norms_sq)
    
    # Distances
    # ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u.v
    # We need ||u - v||^2 >= R_sq
    # 2 u.v <= ||u||^2 + ||v||^2 - R_sq
    
    k = len(all_vectors)
    # For k ~ 1000, k^2 is 10^6. Fine.
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
        # Fallback to just Type A if verification fails (should not happen)
        return type_a_arr
        
    return all_vectors
