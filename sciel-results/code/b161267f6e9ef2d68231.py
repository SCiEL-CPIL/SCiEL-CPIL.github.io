import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    D = 11
    
    # 1. Generate Candidate Pool: Weight-4 vectors
    # Norm^2 = 4. Max norm in set is 2.
    # Constraint: ||v_i - v_j||^2 >= 4 for all i != j.
    # For weight-4 vectors, this means dot(v_i, v_j) <= 2.
    
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
    c_int = candidates.astype(np.int32)
    dot = c_int @ c_int.T
    
    # Conflict if dot > 2
    conflict = (dot > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    # Precompute conflict neighbor lists for fast updates
    # For each node, store indices of its neighbors in the conflict graph
    conflict_neighbors = [np.where(conflict[i] == 1)[0] for i in range(n)]
    
    best_count = 0
    best_selected = np.zeros(n, dtype=np.uint8)
    
    rng = np.random.default_rng(42)
    
    def greedy_construct(temperature=0.0):
        """Greedy min-degree construction with optional randomness."""
        selected = np.zeros(n, dtype=np.uint8)
        remaining = np.ones(n, dtype=np.uint8)
        
        count = 0
        while True:
            if time.time() > deadline - 10.0:
                break
                
            rem_idx = np.where(remaining > 0)[0]
            if len(rem_idx) == 0:
                break
                
            # Calculate effective degree in the remaining subgraph
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n + 1
            
            min_deg = deg.min()
            if min_deg >= n:
                break
                
            if temperature > 0:
                # Soft-min: pick from low-degree nodes with some randomness
                # Use exponential weighting based on degree
                active_deg = deg.copy()
                active_deg[remaining == 0] = min_deg + 100  # large penalty
                # Normalize degrees to [0, 1] range for the active set
                active_mask = remaining > 0
                if active_mask.any():
                    min_active = active_deg[active_mask].min()
                    max_active = active_deg[active_mask].max()
                    if max_active > min_active:
                        scaled = (active_deg - min_active) / (max_active - min_active)
                    else:
                        scaled = np.zeros_like(active_deg)
                    # Exponential weights: lower degree -> higher weight
                    weights = np.exp(-scaled * temperature * 5.0)
                    weights[~active_mask] = 0
                    total_w = weights.sum()
                    if total_w > 0:
                        probs = weights / total_w
                        pick_idx = rng.choice(n, p=probs)
                    else:
                        candidates_to_pick = np.where((deg == min_deg) & (remaining > 0))[0]
                        pick_idx = rng.choice(candidates_to_pick)
                else:
                    break
            else:
                candidates_to_pick = np.where((deg == min_deg) & (remaining > 0))[0]
                if len(candidates_to_pick) == 0:
                    break
                pick_idx = rng.choice(candidates_to_pick)
            
            selected[pick_idx] = 1
            remaining[pick_idx] = 0
            remaining[conflict[pick_idx] == 1] = 0
            count += 1
            
        return selected

    def local_search(selected):
        """Local search with incremental incompat tracking and fast pair finding."""
        sel_indices = np.where(selected == 1)[0]
        if len(sel_indices) == 0:
            return selected, 0
            
        # Initialize incompat: for each node, count conflicts with selected set
        incompat = np.zeros(n, dtype=np.int32)
        for v in sel_indices:
            incompat[conflict_neighbors[v]] += 1
        
        improved = True
        max_moves = 800
        move_count = 0
        moves_made = 0
        
        while improved and move_count < max_moves and time.time() < deadline - 5.0:
            improved = False
            move_count += 1
            
            sel_indices = np.where(selected == 1)[0]
            k_sel = len(sel_indices)
            
            if k_sel == 0:
                break
            
            # Try 1-for-2 moves
            order = rng.permutation(k_sel)
            
            for idx in order:
                if time.time() > deadline - 5.0:
                    break
                    
                v_idx_in_S = idx
                v = sel_indices[v_idx_in_S]
                
                # Candidates u such that u is compatible with S \ {v}
                # u is compatible with S \ {v} if:
                # 1. incompat[u] == 0 (compatible with all S)
                # 2. incompat[u] == 1 and conflict[u, v] == 1 (incompatible only with v)
                
                mask_in_S = selected == 1
                
                cond_0 = (incompat == 0)
                cond_1 = (incompat == 1) & (conflict[:, v] == 1)
                Cand_v = (cond_0 | cond_1) & (~mask_in_S)
                
                cand_indices = np.where(Cand_v > 0)[0]
                
                if len(cand_indices) < 2:
                    continue
                
                # Fast pair finding: for each candidate u, check if any other candidate w
                # is compatible with u. Use conflict rows restricted to cand_indices.
                # Instead of materializing full sub-matrix, iterate and check.
                
                # Optimization: for each u in cand_indices, count conflicts with other cand_indices
                # We want to find any pair (u, w) where conflict[u, w] == 0
                
                # Use a more efficient approach: for each u, check if there exists w != u
                # in cand_indices with conflict[u, w] == 0
                
                found_pair = False
                u_val = -1
                w_val = -1
                
                # Try a sampling approach first for speed, then exhaustive if needed
                # Sample up to 200 pairs randomly
                if len(cand_indices) <= 200:
                    # Small enough to check more thoroughly
                    # For each u, check conflicts with all other candidates
                    for i_u in range(len(cand_indices)):
                        u = cand_indices[i_u]
                        # Check if u has any compatible partner in cand_indices
                        # conflict[u, cand_indices] gives conflict values
                        conflicts_with_cands = conflict[u, cand_indices]
                        # Find positions where conflict is 0 (compatible)
                        compat_positions = np.where(conflicts_with_cands == 0)[0]
                        # Exclude self (position i_u)
                        compat_positions = compat_positions[compat_positions != i_u]
                        if len(compat_positions) > 0:
                            w = cand_indices[compat_positions[0]]
                            u_val = u
                            w_val = w
                            found_pair = True
                            break
                else:
                    # Large candidate set: use random sampling
                    num_samples = min(500, len(cand_indices) * 2)
                    for _ in range(num_samples):
                        i_u = rng.integers(len(cand_indices))
                        i_w = rng.integers(len(cand_indices))
                        if i_u == i_w:
                            continue
                        u = cand_indices[i_u]
                        w = cand_indices[i_w]
                        if conflict[u, w] == 0:
                            u_val = u
                            w_val = w
                            found_pair = True
                            break
                    
                    # If not found by sampling, try a more systematic approach
                    if not found_pair:
                        # Check first 100 candidates against all others
                        check_range = min(100, len(cand_indices))
                        for i_u in range(check_range):
                            u = cand_indices[i_u]
                            conflicts_with_cands = conflict[u, cand_indices]
                            compat_positions = np.where(conflicts_with_cands == 0)[0]
                            compat_positions = compat_positions[compat_positions != i_u]
                            if len(compat_positions) > 0:
                                w = cand_indices[compat_positions[0]]
                                u_val = u
                                w_val = w
                                found_pair = True
                                break
                
                if found_pair:
                    u = u_val
                    w = w_val
                    
                    # Perform move: remove v, add u, w
                    # Update incompat incrementally
                    incompat[conflict_neighbors[v]] -= 1
                    incompat[conflict_neighbors[u]] += 1
                    incompat[conflict_neighbors[w]] += 1
                    
                    selected[v] = 0
                    selected[u] = 1
                    selected[w] = 1
                    improved = True
                    moves_made += 1
                    break  # Restart while loop with new S
            
            # If no 1-for-2 found, try 1-for-1 swaps to escape local optima
            if not improved:
                if k_sel > 0:
                    # Try multiple random swaps
                    for _ in range(5):
                        v_idx_in_S = rng.integers(k_sel)
                        v = sel_indices[v_idx_in_S]
                        
                        cond_0 = (incompat == 0)
                        cond_1 = (incompat == 1) & (conflict[:, v] == 1)
                        Cand_v = (cond_0 | cond_1) & (selected == 0)
                        
                        cand_indices = np.where(Cand_v > 0)[0]
                        
                        if len(cand_indices) > 0:
                            u = cand_indices[rng.integers(len(cand_indices))]
                            
                            # Verify u is actually compatible with S\{v}
                            # (should be by construction, but double-check for safety)
                            # Actually, let's just do the swap and rely on the structure
                            
                            # Update incompat
                            incompat[conflict_neighbors[v]] -= 1
                            incompat[conflict_neighbors[u]] += 1
                            
                            selected[v] = 0
                            selected[u] = 1
                            improved = True
                            moves_made += 1
                            break
                            # Don't break outer loop - continue searching for 1-for-2
        
        return selected, moves_made

    # Multi-start search with varying temperatures
    num_restarts = 1000  # Will be limited by time
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Vary temperature for diversity
        if restart < 50:
            temp = 0.0  # Pure greedy for first few
        elif restart < 200:
            temp = 0.5  # Mild randomness
        elif restart < 500:
            temp = 1.0  # Moderate randomness
        else:
            temp = 2.0  # High randomness for exploration
        
        # Phase 1: Greedy
        selected = greedy_construct(temperature=temp)
        
        # Phase 2: Local Search
        selected, moves = local_search(selected)
        
        current_count = selected.sum()
        if current_count > best_count:
            best_count = current_count
            best_selected = selected.copy()
            
            # If we found a good solution, do an extra refinement pass
            if current_count > best_count - 5 and time.time() < deadline - 30.0:
                selected2, _ = local_search(best_selected.copy())
                c2 = selected2.sum()
                if c2 > best_count:
                    best_count = c2
                    best_selected = selected2.copy()

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
