import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a lookahead-based heuristic matrix.
    Heuristic for edge (i, j) is based on the total prize potential
    of visiting j and then greedily visiting other nodes reachable
    within the remaining budget, normalized by the cost of the detour.
    """
    start_time = time.time()
    N = len(prizes)
    eps = 1e-9
    gamma = 1.5 # Distance sensitivity
    
    # Precompute distances from start (node 0)
    d0 = distance_matrix[0]
    
    # Precompute sorted neighbors for each node by distance
    # neighbors[j] = list of (dist_jk, prize_k, k) sorted by dist_jk
    neighbors = []
    for j in range(N):
        # Get distances from j to all k
        djs = distance_matrix[j].copy()
        djs[j] = np.inf # Exclude self
        # Sort indices by distance
        sorted_indices = np.argsort(djs)
        # Create list of (dist, prize, index)
        neigh_list = [(djs[k], prizes[k], k) for k in sorted_indices]
        neighbors.append(neigh_list)
        
    # Initialize heuristic matrix
    H = np.zeros((N, N))
    
    # For each edge (i, j)
    for i in range(N):
        for j in range(N):
            if i == j:
                H[i, j] = eps
                continue
                
            d_ij = distance_matrix[i, j]
            d_j0 = d0[j]
            
            # Cost to go i->j and then back to 0 (minimum commitment)
            cost_commit = d_ij + d_j0
            
            # Approximate remaining budget for further nodes after j
            # Assume the cost to reach i was d0[i] (best case scenario for heuristic estimation)
            # This is an upper bound on available budget, which is safe for potential estimation
            # In reality, path length to i might be > d0[i], so this overestimates potential,
            # but it keeps the heuristic attractive to nodes that *could* be part of a good tour.
            approx_remaining = max_len - d0[i] - cost_commit
            
            if approx_remaining < 0:
                # Even with best-case arrival at i, we can't afford i->j->0
                # Or rather, the commitment cost exceeds max_len
                # Actually, the check in the ant algorithm is:
                # len_so_far + d_ij + d_j0 <= max_len
                # If d0[i] + d_ij + d_j0 > max_len, it's *possible* the ant can still take it
                # if len_so_far < d0[i]. But d0[i] is the minimum possible len_so_far.
                # So if min_possible_len + cost > max_len, it's never valid.
                H[i, j] = eps
                continue
            
            # Calculate potential prize from node j
            # 1. Prize of j itself
            potential = prizes[j]
            
            # 2. Greedy addition of neighbors
            # We want to add prizes of nodes k such that we can visit them after j
            # Constraint for visiting k after j:
            # len_so_far + d_ij + d_jk + d_k0 <= max_len
            # Approximate len_so_far + d_ij as d0[i] (since we used d0[i] in approx_remaining)
            # So condition becomes: d0[i] + d_jk + d_k0 <= max_len
            # => d_jk + d_k0 <= max_len - d0[i]
            # Let limit_k = max_len - d0[i]
            
            limit_k = max_len - d0[i]
            
            for d_jk, prize_k, k in neighbors[j]:
                if d_jk == np.inf:
                    break
                # Check if k can be visited after j
                # The cost to go j->k and then k->0 is d_jk + d_k0
                # We have 'approx_remaining' budget left? No.
                # The total length would be: Path_to_i + d_ij + d_jk + d_k0
                # We approximated Path_to_i as d0[i].
                # So total approx length = d0[i] + d_ij + d_jk + d_k0
                # We need this <= max_len
                # d0[i] + d_ij + d_jk + d_k0 <= max_len
                # d_jk + d_k0 <= max_len - d0[i] - d_ij
                
                # We already checked d_ij + d_j0 <= max_len - d0[i] for the edge to be valid at all?
                # No, we checked d0[i] + d_ij + d_j0 <= max_len.
                # The budget for *additional* nodes k is:
                # remaining_budget_for_k = max_len - d0[i] - d_ij
                # Condition for k: d_jk + d_k0 <= remaining_budget_for_k
                
                rem_budget = max_len - d0[i] - d_ij
                if rem_budget < 0:
                    break # Neighbors are sorted by d_jk, so if first doesn't fit, none will? 
                         # Wait, d_k0 varies. But d_jk is increasing. 
                         # d_jk + d_k0 is not necessarily sorted.
                         # However, d_jk is increasing. d_k0 is bounded.
                         # We just check the condition.
                    pass
                
                if d_jk + d0[k] <= rem_budget:
                    potential += prize_k
                else:
                    # Since neighbors are sorted by d_jk, and d0[k] is relatively constant/small variation,
                    # if d_jk is too large, later ones will be larger.
                    # But d0[k] can be large.
                    # To be safe, don't break, just skip.
                    # Optimization: if d_jk > rem_budget, then d_jk + d0[k] > rem_budget is likely.
                    # We'll just continue.
                    pass
            
            # Heuristic is Potential / Cost^gamma
            # Cost is the "efficiency" cost: d_ij + d_j0
            H[i, j] = potential / (cost_commit ** gamma + eps)
            
            # Deadline check
            if time.time() - start_time > 250:
                # Return what we have so far, padding rest with 1e-9
                # But we need a full matrix. 
                # If we run out of time, we should probably return a simpler heuristic 
                # or just fill the rest with the base ratio.
                # For now, break and fill rest with base ratio
                for i_rem in range(i, N):
                    for j_rem in range(0 if i_rem != i else j, N):
                        if i_rem == j_rem:
                            H[i_rem, j_rem] = eps
                        else:
                            # Fallback to simple ratio
                            d_ij_rem = distance_matrix[i_rem, j_rem]
                            d_j0_rem = d0[j_rem]
                            H[i_rem, j_rem] = prizes[j_rem] / (d_ij_rem + d_j0_rem + eps)
                break
        
        if time.time() - start_time > 250:
            break

    # Ensure non-finite values are handled
    H = np.where(np.isfinite(H), H, eps)
    H = np.where(H > 0, H, eps)
    
    return H
