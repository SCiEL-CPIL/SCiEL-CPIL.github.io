import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate all D_11 vectors with squared norm 4
    candidates = []
    
    # Type 1: One ±2, rest 0
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)
            
    # Type 2: Four ±1's, rest 0
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)
    
    # Compute dot product matrix
    M = candidates @ candidates.T
    
    # Conflict: two vectors are incompatible if M[i,j] > 2
    conflict = (M > 2).astype(np.uint8)
    np.fill_diagonal(conflict, 0)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = None
    
    # Try multiple random restarts
    for restart in range(20):
        if time.time() > deadline - 20:
            break
            
        # Random order
        order = rng.permutation(N)
        
        in_set = np.zeros(N, dtype=bool)
        conflict_counts = np.zeros(N, dtype=np.int32)
        
        # Greedy
        for idx in order:
            if conflict_counts[idx] == 0:
                in_set[idx] = True
                conflict_counts += conflict[idx]
                
        current_size = np.sum(in_set)
        
        # Local Search: 1-for-2 swaps
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Find nodes in ~S with exactly 1 conflict in S
            # conflict_counts[j] == 1
            # Find the unique neighbor in S for each such j
            # This is O(N^2) if done naively.
            # Optimization: 
            # For each i in S, find j in ~S such that conflict[i,j]==1 and conflict_counts[j]==1.
            # We can precompute for each i, the list of j in ~S with conflict_counts[j]==1 and conflict[i,j]==1.
            
            # Let's identify candidates for removal
            # A node i in S is a candidate if it blocks at least 2 nodes j that don't conflict with each other.
            
            # Get indices of nodes with exactly 1 conflict
            blocked_by_one = np.where((~in_set) & (conflict_counts == 1))[0]
            
            if len(blocked_by_one) == 0:
                break
            
            # Map each blocked node to its blocker
            # blocker[j] = i if conflict[i,j]==1 and in_set[i]
            # Since conflict_counts[j]==1, there is exactly one such i.
            
            # We can find this by:
            # For each j in blocked_by_one:
            #   i = np.where(conflict[j] & in_set)[0][0]
            
            # This is slow in Python loop.
            # Vectorized approach:
            # Create a matrix B where B[j, i] = 1 if conflict[j,i] and in_set[i].
            # Sum over i? No.
            
            # Let's just iterate through i in S and check how many blocked_by_one nodes it blocks.
            # blocked_by_one is a list of indices.
            # For each i in S:
            #   conflicts = conflict[i, blocked_by_one]
            #   count = sum(conflicts)
            #   if count >= 2:
            #       candidates_to_add = blocked_by_one[conflicts == 1]
            #       Check if candidates_to_add contains an independent set of size >= 2.
            
            selected_idx = np.where(in_set)[0]
            
            # To speed up, only check i that block many nodes?
            # Or just iterate.
            
            for i in selected_idx:
                if time.time() > deadline - 10:
                    break
                    
                # Find which blocked_by_one nodes are blocked by i
                # conflict[i, blocked_by_one]
                mask = conflict[i, blocked_by_one]
                if np.sum(mask) < 2:
                    continue
                    
                addable_candidates = blocked_by_one[mask]
                
                # Check if we can pick 2 non-conflicting nodes from addable_candidates
                # Since addable_candidates are all blocked ONLY by i, they don't conflict with S \ {i}.
                # But they might conflict with each other.
                # We need an independent set of size 2 in the subgraph induced by addable_candidates.
                # This is just checking if there exists a pair (u, v) in addable_candidates such that conflict[u, v] == 0.
                
                # If len(addable_candidates) == 2, check if they conflict.
                # If len > 2, check if ANY pair is non-conflicting.
                # Actually, if the subgraph is not a clique, we can pick 2.
                # A clique of size k has all conflicts.
                # So we just need to find one non-conflict.
                
                found_pair = False
                n_cand = len(addable_candidates)
                
                if n_cand == 2:
                    u, v = addable_candidates
                    if conflict[u, v] == 0:
                        found_pair = True
                        pair = [u, v]
                else:
                    # Check all pairs? O(k^2). k is likely small.
                    # If k is large, it's very likely there is a non-conflict.
                    # Just check a few random pairs or all if small.
                    for a_idx in range(n_cand):
                        for b_idx in range(a_idx + 1, n_cand):
                            u = addable_candidates[a_idx]
                            v = addable_candidates[b_idx]
                            if conflict[u, v] == 0:
                                found_pair = True
                                pair = [u, v]
                                break
                        if found_pair:
                            break
                            
                if found_pair:
                    # Remove i
                    in_set[i] = False
                    conflict_counts -= conflict[i]
                    current_size -= 1
                    
                    # Add pair
                    in_set[pair[0]] = True
                    conflict_counts += conflict[pair[0]]
                    in_set[pair[1]] = True
                    conflict_counts += conflict[pair[1]]
                    current_size += 2
                    
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set.copy()
                        best_conflict_counts = conflict_counts.copy()
                    # Restart local search? Or continue?
                    # Continue is fine, but blocked_by_one list is stale.
                    # Break inner loop to recompute blocked_by_one?
                    # Yes, better to recompute.
                    break
                    
            if improved:
                # Recompute blocked_by_one for next iteration
                continue
            else:
                break
                
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
            best_conflict_counts = conflict_counts.copy()

    # Finalize
    if best_size == 0 or best_in_set is None:
        return candidates[:22].astype(np.int64)
        
    result = candidates[best_in_set].astype(np.int64)
    
    # Verify
    n = len(result)
    if n == 0:
        return candidates[:22].astype(np.int64)
        
    max_norm_sq = int(np.max(np.sum(result**2, axis=1)))
    diffs = result[:, np.newaxis, :] - result[np.newaxis, :, :]
    diff_norms_sq = np.sum(diffs**2, axis=2)
    if np.min(diff_norms_sq[np.triu_indices(n, k=1)]) < max_norm_sq:
        return candidates[:22].astype(np.int64)
        
    return result
