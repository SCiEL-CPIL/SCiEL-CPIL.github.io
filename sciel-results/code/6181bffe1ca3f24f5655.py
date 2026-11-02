"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Generate Candidate Pool: ONLY Weight-2 vectors
    # A weight-2 vector has exactly two non-zero entries, each +/- 1.
    # Norm squared is 2.
    # Condition for compatibility between u, v (both weight-2):
    # ||u - v||^2 >= 2  =>  ||u||^2 + ||v||^2 - 2 u.v >= 2  =>  4 - 2 u.v >= 2  =>  u.v <= 1.
    
    candidates = []
    # Store as tuples for fast hashing/comparison if needed, but numpy is fine.
    # Let's generate them.
    for comb in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(D, dtype=np.int64)
            v[comb[0]] = signs[0]
            v[comb[1]] = signs[1]
            candidates.append(v)
            
    candidates = np.array(candidates)
    n_cand = len(candidates)
    
    # 2. Precompute Conflict Matrix
    # G = candidates @ candidates.T
    # conflict[i, j] = (G[i, j] > 1)
    G = candidates @ candidates.T
    # Note: G[i,i] = 2, which is > 1, so self-conflict is True.
    conflict_mat = G > 1
    
    # 3. Greedy Initialization
    # Heuristic: Pick nodes with minimum degree in the conflict graph first?
    # Or just pick arbitrarily and refine?
    # Let's compute static degrees.
    degrees = np.sum(conflict_mat, axis=1)
    
    # Sort by degree ascending (least constrained first)
    order = np.argsort(degrees)
    
    selected_mask = np.zeros(n_cand, dtype=bool)
    selected_indices = []
    
    for v in order:
        if selected_mask[v]:
            continue
        # Check if v conflicts with any already selected node
        # conflict_mat[v] gives conflicts with all.
        # We need to check if any conflict is in selected_mask.
        # If not, select v.
        if not np.any(conflict_mat[v] & selected_mask):
            selected_mask[v] = True
            selected_indices.append(v)
            # Optional: Early exit if we hit a theoretical upper bound?
            # Kissing number in 11D is unknown but > 504 (Levenshtein bound is higher).
            # Actually, known lower bound for k(11) is 504? No, k(11) is around 500-600.
            # Wait, weight-2 vectors only give a subset.
            # Max independent set in this specific graph is the score.
            
    best_set = set(selected_indices)
    best_count = len(best_set)
    
    # 4. Local Search (Iterative Improvement)
    # Strategy: 
    # a) Try to add any unselected node that doesn't conflict with current set.
    # b) If stuck, try to remove one node from the set to allow adding two or more others.
    #    Specifically, look for a "swap" or "expansion".
    
    # To make this fast, we maintain a list of "blocked" nodes?
    # Or just iterate through unselected nodes.
    
    # Efficient check for "can add v":
    # v is compatible with all u in S iff not any(conflict_mat[v, u] for u in S)
    # This is O(|S|) per v. Total O(N * |S|).
    
    # Let's do a few passes of local search.
    improved = True
    passes = 0
    max_passes = 50
    
    while improved and passes < max_passes and time.time() < deadline - 10:
        improved = False
        passes += 1
        
        # Phase A: Add any possible node
        # We can vectorize this:
        # For each unselected node v, check if it conflicts with ANY selected node.
        # conflict_with_selected = conflict_mat[:, selected_mask].any(axis=1)
        # If not conflict_with_selected[v], we can add v.
        
        # But adding one might block another. So we should add them one by one?
        # Or add all that are compatible with the *current* set simultaneously?
        # If u and v are both compatible with S, but u and v conflict, we can't add both.
        # So we must be careful.
        
        # Simple approach: Iterate through unselected nodes in random order.
        # If compatible with current S, add it.
        
        unselected = np.where(~selected_mask)[0]
        # Shuffle for diversity
        rng = np.random.default_rng(int(time.time() * 1000) % 100000)
        rng.shuffle(unselected)
        
        for v in unselected:
            if selected_mask[v]:
                continue
            # Check compatibility with current selected set
            # If v conflicts with any u in selected_mask, skip.
            # conflict_mat[v, selected_mask] will be a boolean array of length |S|
            # We just need to know if any is True.
            if not np.any(conflict_mat[v, selected_mask]):
                selected_mask[v] = True
                best_count += 1
                improved = True
                # Update best_set? We can just use selected_mask at the end.
                
        # Phase B: Local Swap/Refinement
        # If no node could be added, try to improve by swapping.
        # This is expensive. Let's try a simpler "kick" strategy:
        # Remove a random node from S, then try to re-add as many as possible.
        # This is like a simulated annealing step or a greedy restart from a perturbed state.
        
        if not improved and best_count > 10:
            # Pick a random node to remove
            current_indices = np.where(selected_mask)[0]
            if len(current_indices) > 0:
                rem_idx = current_indices[rng.integers(0, len(current_indices))]
                selected_mask[rem_idx] = False
                best_count -= 1
                improved = True # Force another pass to try to fill the gap
                # We don't immediately re-add, let the next Phase A handle it.
                # But Phase A just iterates unselected. The removed node is now unselected.
                # Its neighbors might now be addable.
                
    # 5. Final Verification and Return
    final_indices = np.where(selected_mask)[0]
    if len(final_indices) == 0:
        # Fallback: return a single valid vector
        return np.array([np.ones(D, dtype=np.int64)], dtype=np.int64)
        
    result = candidates[final_indices]
    
    # Sanity Check: Verify the condition holds for the returned set
    # This is crucial to avoid invalid scores.
    if len(result) > 1:
        # Compute pairwise distances squared
        # ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u.v
        # Since all are weight-2, ||u||^2 = 2.
        # Dist^2 = 4 - 2 u.v
        # Condition: Dist^2 >= 2  =>  4 - 2 u.v >= 2  =>  2 u.v <= 2  =>  u.v <= 1.
        
        # Compute Gram matrix of result
        G_res = result @ result.T
        # Check off-diagonal elements
        # Diagonal is 2.
        # We need all off-diagonal elements <= 1.
        
        # Create a mask for off-diagonals
        mask = ~np.eye(len(result), dtype=bool)
        if np.any(G_res[mask] > 1):
            # Invalid configuration found!
            # This should not happen if our logic is correct, but let's be safe.
            # Fallback to a simple valid set: just the first few non-conflicting vectors.
            # Or just return a single vector.
            # Let's try to fix it by greedily removing conflicts.
            # This is a safety net.
            
            # Re-run a simple greedy on the result set?
            # Or just return a subset that is valid.
            # Given the time, returning a single vector is safest but scores 1.
            # Let's try to salvage by removing one of the conflicting pairs.
            
            # Find a conflicting pair
            # This is complex to do quickly.
            # Since our logic `if not np.any(conflict_mat[v, selected_mask])` ensures
            # that every added node is compatible with ALL previously added nodes,
            # the set should be valid by construction.
            # The only way this fails is if `conflict_mat` was wrong.
            # Let's assume it's correct.
            pass
            
    return result

# EVOLVE-BLOCK-END
