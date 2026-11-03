import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    # Generate all weight-4 vectors: C(11,4)*16 = 330*16 = 5280
    # Norm squared is 4.
    # Condition: min pairwise distance >= max norm.
    # If we restrict to weight-4 vectors, max norm = 2.
    # Min pairwise distance >= 2 => min pairwise distance squared >= 4.
    # ||v - w||^2 = ||v||^2 + ||w||^2 - 2*v.w = 4 + 4 - 2*v.w = 8 - 2*v.w.
    # 8 - 2*v.w >= 4 => 4 >= 2*v.w => v.w <= 2.
    # So conflict if dot product > 2.
    
    candidates_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
    
    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)
    
    # Precompute Conflict Matrix
    # conflict[i, j] = 1 if dot(candidates[i], candidates[j]) > 2
    c_int = candidates.astype(np.int32)
    dot = c_int @ c_int.T
    conflict = (dot > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    # Compatibility matrix (1 if compatible)
    compat = 1 - conflict
    np.fill_diagonal(compat, 1)
    
    # Precompute degrees for greedy heuristic
    degrees = conflict.sum(axis=1)
    
    best_count = 0
    best_selected = np.zeros(n, dtype=np.uint8)
    
    rng = np.random.default_rng(42)
    
    # Multi-Start Search
    # We will run several iterations of:
    # a) Randomized Greedy Construction
    # b) Local Search Refinement (1-for-2 swaps)
    
    num_restarts = 200
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
            
        # --- Phase 1: Randomized Greedy Construction ---
        selected = np.zeros(n, dtype=np.uint8)
        remaining = np.ones(n, dtype=np.uint8)
        
        count = 0
        # To speed up, we can maintain a list of active nodes or just use masks.
        # With N=5280, full vector operations are fast.
        
        while True:
            if time.time() > deadline - 10.0:
                break
                
            # Find remaining vertices
            rem_idx = np.where(remaining > 0)[0]
            if len(rem_idx) == 0:
                break
                
            # Calculate effective degree in the remaining subgraph
            # deg[i] = sum(conflict[i, j] for j in remaining)
            # We can compute this vectorized: conflict[:, rem_idx].sum(axis=1)
            # But this is O(N * |rem|). 
            # Optimization: Maintain a running "compat_with_selected" mask?
            # Actually, the degree in the remaining graph is what matters for Min-Degree.
            # Let's just compute it. It's fast enough for 5280.
            
            # deg[i] = (conflict[i, :] * remaining).sum()
            # Vectorized:
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[remaining == 0] = n + 1 # Mask out non-remaining
            
            min_deg = deg.min()
            if min_deg >= n:
                break
                
            # Pick randomly from those with min degree
            candidates_to_pick = np.where((deg == min_deg) & (remaining > 0))[0]
            if len(candidates_to_pick) == 0:
                break
                
            # Add some randomness: pick from top 10% min-degree nodes?
            # Or just pick randomly from the min-degree set.
            # Let's pick randomly from the min-degree set.
            pick_idx = rng.choice(candidates_to_pick)
            
            selected[pick_idx] = 1
            remaining[pick_idx] = 0
            # Update remaining: remove neighbors of pick_idx
            remaining[conflict[pick_idx] == 1] = 0
            count += 1
            
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
            
        # --- Phase 2: Local Search Refinement ---
        # We will try to improve the set size using 1-for-2 swaps.
        # Remove v, add u, w.
        
        sel_indices = np.where(selected == 1)[0]
        if len(sel_indices) == 0:
            continue
            
        improved = True
        max_moves = 20
        move_count = 0
        
        while improved and move_count < max_moves and time.time() < deadline - 5.0:
            improved = False
            move_count += 1
            
            # Shuffle selected indices to avoid bias
            order = rng.permutation(len(sel_indices))
            
            for i in order:
                if time.time() > deadline - 5.0:
                    break
                    
                v = sel_indices[i]
                
                # Compute mask of vertices compatible with (S \ {v})
                # mask_except_v[j] = 1 if j is compatible with all s in S, s != v
                mask_except_v = np.ones(n, dtype=np.uint8)
                for s in sel_indices:
                    if s != v:
                        mask_except_v = mask_except_v & compat[s]
                
                # Candidates for addition must be:
                # 1. Not in S
                # 2. Compatible with S \ {v}
                mask_except_v[sel_indices] = 0
                
                cand_indices = np.where(mask_except_v > 0)[0]
                if len(cand_indices) < 2:
                    continue
                    
                # We need to find two candidates u, w from cand_indices
                # such that u and w are compatible with each other.
                # This is finding an edge in the subgraph induced by cand_indices.
                
                # Subgraph conflict matrix for candidates
                # This can be expensive if cand_indices is large.
                # But cand_indices is usually small because S is large.
                # Let's try.
                
                cand_conflict = conflict[np.ix_(cand_indices, cand_indices)]
                
                # Find any pair (u, w) with no conflict
                # We can just iterate through a few random pairs or find the first one.
                # To be fast, we can check if there is any 0 in the upper triangle.
                
                # Get upper triangle indices
                rows, cols = np.where(cand_conflict == 0)
                # Exclude diagonal
                mask_off_diag = rows != cols
                rows = rows[mask_off_diag]
                cols = cols[mask_off_diag]
                
                if len(rows) == 0:
                    continue
                    
                # Pick a random pair
                pair_idx = rng.integers(0, len(rows))
                u_local = rows[pair_idx]
                w_local = cols[pair_idx]
                
                u_global = cand_indices[u_local]
                w_global = cand_indices[w_local]
                
                # Perform swap
                selected[v] = 0
                selected[u_global] = 1
                selected[w_global] = 1
                count += 1
                improved = True
                
                # Update sel_indices
                sel_indices = np.where(selected == 1)[0]
                
                # Update best if improved
                if count > best_count:
                    best_count = count
                    best_selected = selected.copy()
                
                # Break out of inner loop to restart the while loop with new state
                break
        
    # Build result
    if best_count == 0:
        # Fallback
        result = np.zeros((1, D), dtype=np.int64)
        result[0, 0] = 1
        result[0, 1] = 1
        result[0, 2] = 1
        result[0, 3] = 1
        return result

    result = np.zeros((best_count, D), dtype=np.int64)
    sel_indices = np.where(best_selected == 1)[0]
    for i, idx in enumerate(sel_indices):
        result[i] = candidates[idx]
        
    return result

