"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Generate Candidate Pool: Weight-4 vectors
    # Norm^2 = 4. Max norm in set is 2.
    # Constraint: ||v_i - v_j||^2 >= 4 for all i != j.
    # For weight-4 vectors, this means dot(v_i, v_j) <= 2.
    # dist^2 = 4 + 4 - 2*dot = 8 - 2*dot.
    # dist^2 >= 4  =>  8 - 2*dot >= 4  =>  4 >= 2*dot  =>  dot <= 2.
    
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
    # We primarily use conflict for updates.
    
    best_count = 0
    best_selected = np.zeros(n, dtype=np.uint8)
    
    # 3. Multi-Start Search
    # Strategy:
    # A) Randomized Greedy Construction (Min-Degree heuristic)
    # B) Vectorized Local Search Refinement (1-for-2 swaps primarily)
    
    rng = np.random.default_rng(42)
    
    def greedy_construct():
        """
        Greedy selection using minimum-degree heuristic on the remaining graph.
        Returns a binary vector indicating selected nodes.
        """
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
            # deg[i] = number of conflicts for i among remaining nodes
            # Using matrix-vector product: conflict @ remaining
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            
            # Only consider remaining nodes
            deg[remaining == 0] = n + 1
            
            min_deg = deg.min()
            if min_deg >= n:
                break
                
            # Pick randomly from those with min degree to add diversity
            candidates_to_pick = np.where((deg == min_deg) & (remaining > 0))[0]
            if len(candidates_to_pick) == 0:
                break
                
            pick_idx = rng.choice(candidates_to_pick)
            
            selected[pick_idx] = 1
            remaining[pick_idx] = 0
            # Update remaining: remove neighbors of pick_idx
            remaining[conflict[pick_idx] == 1] = 0
            count += 1
            
        return selected

    def local_search(selected):
        """
        Refine the selected set using 1-for-2 moves.
        A 1-for-2 move removes one vector v and adds two vectors u, w
        such that u, w are compatible with each other and with S\{v}.
        This increases the set size by 1.
        """
        sel_indices = np.where(selected == 1)[0]
        if len(sel_indices) == 0:
            return selected, 0
            
        improved = True
        max_moves = 200
        move_count = 0
        moves_made = 0
        
        while improved and move_count < max_moves and time.time() < deadline - 5.0:
            improved = False
            move_count += 1
            
            sel_indices = np.where(selected == 1)[0]
            k_sel = len(sel_indices)
            
            if k_sel == 0:
                break
            
            # Compute Total_Incompatibility for all nodes against current S
            # conflict[:, sel_indices] gives incompatibility matrix for S
            # We want to know, for each node x, how many selected nodes it conflicts with.
            conflict_with_S = conflict[:, sel_indices]
            total_inc = conflict_with_S.sum(axis=1)
            
            # base_mask: nodes compatible with ALL S (total_inc == 0) and not selected
            base_mask = (total_inc == 0) & (selected == 0)
            
            # Try 1-for-2 moves
            # For each v in S, find u, w compatible with S\{v} and with each other
            # We iterate through S in a random order to avoid bias
            order = rng.permutation(k_sel)
            
            for idx in order:
                if time.time() > deadline - 5.0:
                    break
                    
                v_idx_in_S = idx
                v = sel_indices[v_idx_in_S]
                
                # Construct Cand_v: nodes compatible with S \ {v}
                # A node x is compatible with S\{v} if:
                # 1. It is compatible with all of S (total_inc[x] == 0), OR
                # 2. It conflicts ONLY with v (total_inc[x] == 1 and conflict[x, v] == 1)
                # And x is not currently selected.
                
                cond_mask = (total_inc == 1) & (conflict_with_S[:, v_idx_in_S] == 1) & (selected == 0)
                Cand_v = base_mask | cond_mask
                
                cand_indices = np.where(Cand_v > 0)[0]
                
                if len(cand_indices) < 2:
                    continue
                
                # Check if there is any compatible pair in cand_indices
                # sub_conflict[u, w] == 0 means compatible
                sub_conflict = conflict[np.ix_(cand_indices, cand_indices)]
                
                # We need off-diagonal 0s
                # Mask diagonal to 1 to ignore self-compatibility
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
                        break # Restart while loop with new S
                        
            # If no 1-for-2 found, we stop to save time.
            # Could add 1-for-1 swaps for better exploration, but 1-for-2 is primary.
            
        return selected, moves_made

    # Run multiple restarts
    # Estimate time per restart.
    # Greedy is O(K * N) roughly, Local Search is O(Moves * K^2 + Moves * N^2) roughly.
    # Let's run a few restarts to see how long they take.
    
    max_restarts = 50
    restart_count = 0
    
    for i in range(max_restarts):
        if time.time() > deadline - 20.0:
            break
            
        # Greedy construction
        selected = greedy_construct()
        current_count = selected.sum()
        
        # Local search refinement
        selected, moves = local_search(selected)
        current_count = selected.sum()
        
        if current_count > best_count:
            best_count = current_count
            best_selected = selected.copy()
            
        # If we've found a very good solution, we might want to spend more time refining it?
        # For now, just keep going with new restarts.
        
        restart_count += 1
    
    # If we have time left, try to refine the best solution further
    if time.time() < deadline - 5.0 and best_count > 0:
        # Run local search again on the best solution with more moves allowed?
        # Or just return it.
        pass

    result = candidates[best_selected == 1]
    
    # Fallback if nothing was selected (should not happen)
    if len(result) == 0:
         # Return a single valid vector
         v = np.zeros(D, dtype=np.int64)
         v[0] = 1
         v[1] = 1
         v[2] = 1
         v[3] = 1
         return v.reshape(1, -1)

    return result

# EVOLVE-BLOCK-END
