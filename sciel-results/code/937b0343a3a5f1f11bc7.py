def construct():
    import time
    from math import gcd, lcm
    from fractions import Fraction
    import sys

    start_time = time.time()
    # Use 55s deadline to be safe within 60s/90s limits
    deadline = start_time + 55

    N_MAX = 256
    # We consider values 2..256
    values = list(range(2, N_MAX + 1))
    N = len(values)
    
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(values)}
    
    # Weights are 1/v. Using Fraction for exact comparison in critical path?
    # Floating point is likely sufficient for ranking, but let's use Fraction for the final score verification if needed.
    # For speed, use float for bounds and search.
    weights = [1.0 / v for v in values]

    # Build conflict graph (incompatibility).
    # Edge (i, j) exists if lcm(values[i], values[j]) <= 256.
    # We want an Independent Set in this conflict graph, which is a Clique in the compatibility graph.
    # Let's work with the Conflict Graph and find Maximum Weight Independent Set (MWIS).
    
    # adj_conflict[i] is a bitmask of j such that i and j are INCOMPATIBLE (lcm <= 256).
    # i and j are compatible if NOT (adj_conflict[i] & (1 << j)).
    
    adj_conflict = [0] * N
    for i in range(N):
        vi = values[i]
        for j in range(i + 1, N):
            vj = values[j]
            if lcm(vi, vj) <= 256:
                adj_conflict[i] |= (1 << j)
                adj_conflict[j] |= (1 << i)

    # Helper to get list of indices from mask
    def mask_to_list(mask):
        res = []
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            res.append(idx)
            mask ^= lsb
        return res

    # Helper to sum weights of mask
    def sum_weights(mask):
        s = 0.0
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            s += weights[idx]
            mask ^= lsb
        return s

    # Greedy initial solution for MWIS (Independent Set in conflict graph)
    # Greedy: Pick node with max weight, remove it and its neighbors (conflicts).
    # This yields a valid independent set.
    available = [True] * N
    R_greedy_mask = 0
    sum_greedy = 0.0
    # To be faster, we can just iterate.
    # Since N=255, O(N^2) is fine.
    for _ in range(N):
        best_idx = -1
        best_w = -1.0
        for i in range(N):
            if available[i] and weights[i] > best_w:
                best_w = weights[i]
                best_idx = i
        if best_idx == -1:
            break
        R_greedy_mask |= (1 << best_idx)
        sum_greedy += best_w
        # Remove best_idx and its conflicts from available
        available[best_idx] = False
        # Conflicts of best_idx are bits set in adj_conflict[best_idx]
        conflicts = adj_conflict[best_idx]
        while conflicts:
            lsb = conflicts & -conflicts
            idx = lsb.bit_length() - 1
            available[idx] = False
            conflicts ^= lsb

    best_sum = sum_greedy
    best_set_mask = R_greedy_mask

    # Bron-Kerbosch for Maximum Weight Independent Set
    # We can view this as finding a Max Weight Clique in the Compatibility Graph.
    # Or directly MWIS on Conflict Graph.
    # Let's use BK on the Conflict Graph to find Independent Sets.
    # R: current independent set
    # P: candidates that can be added to R (must be non-adjacent to all in R)
    # X: candidates that were already considered (to avoid duplicates)
    
    # In Conflict Graph:
    # v can be added to R if v is not adjacent to any node in R.
    # So P = { v | v is not adjacent to any u in R }.
    # This is equivalent to P = { v | adj_conflict[v] & R_mask == 0 }.
    
    # BK recursion:
    # 1. If P and X are empty, update best.
    # 2. Bound: current_sum + sum(weights in P) <= best_sum -> prune.
    # 3. Pivot selection: Choose u in P ∪ X that maximizes |P ∩ N(u)| in Conflict Graph.
    #    Wait, standard BK for Clique uses N(v) in the graph.
    #    For Independent Set in Conflict Graph, the "neighbors" in the compatibility sense are non-neighbors in conflict graph.
    #    Let's stick to standard BK on the Compatibility Graph (Clique).
    #    Compatibility Graph Edge: lcm > 256.
    #    Let's build adj_compat.
    
    adj_compat = [0] * N
    for i in range(N):
        # adj_compat[i] = ~adj_conflict[i] & all_nodes_mask (excluding self?)
        # In BK, we usually don't include self in N(v).
        # Let's construct adj_compat explicitly.
        pass
    
    # Rebuilding adj_compat for clarity and speed in BK
    all_nodes_mask = (1 << N) - 1
    for i in range(N):
        # Start with all, remove conflicts, remove self
        adj_compat[i] = all_nodes_mask ^ (1 << i) ^ adj_conflict[i]

    def bk(R_mask, P_mask, X_mask, current_sum):
        nonlocal best_sum, best_set_mask

        if not P_mask:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set_mask = R_mask
            return

        # Bound
        # Sum of weights in P
        # Optimization: Precompute prefix sums? No, mask is dynamic.
        # Just sum.
        bound_val = sum_weights(P_mask)
        if current_sum + bound_val <= best_sum:
            return

        # Pivot selection
        # Choose u in P ∪ X maximizing |P ∩ N(u)| in Compat Graph.
        # This minimizes the number of candidates.
        max_deg = -1
        pivot = -1
        
        # Check P first, then X? Or just P is often good enough.
        # Standard is P ∪ X.
        # For speed, let's just check P. If P is empty, we returned.
        temp_P = P_mask
        while temp_P:
            lsb = temp_P & -temp_P
            u = lsb.bit_length() - 1
            deg = (P_mask & adj_compat[u]).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot = u
            temp_P ^= lsb
        
        # Also check X? X can be large. 
        # If we only check P, we might miss a better pivot.
        # But checking X adds overhead. Let's stick to P for now, or limit X check.
        # Actually, for small N, checking P is fast.
        
        if pivot == -1:
            # P is empty? Handled above.
            return

        # Candidates: P \ N(pivot)
        # In Compat Graph, N(pivot) are neighbors.
        # We iterate over v in P that are NOT neighbors of pivot.
        candidates = P_mask & ~adj_compat[pivot]
        
        # If candidates is empty, it means all in P are neighbors of pivot.
        # But pivot is in P or X.
        # If pivot in P, then pivot is not in N(pivot) (no self loop), so pivot is in candidates.
        # If pivot in X, then candidates could be empty?
        # If P subset of N(pivot), then no v in P is non-neighbor.
        # But every maximal clique contains a vertex from P \ N(pivot).
        # If P \ N(pivot) is empty, then no new maximal clique can be formed?
        # Actually, if P \ N(pivot) is empty, the loop does nothing.
        
        temp_C = candidates
        while temp_C:
            lsb = temp_C & -temp_C
            v = lsb.bit_length() - 1
            
            # Recurse
            new_R = R_mask | lsb
            # New P: neighbors of v in P
            new_P = P_mask & adj_compat[v]
            # New X: neighbors of v in X
            new_X = X_mask & adj_compat[v]
            
            bk(new_R, new_P, new_X, current_sum + weights[v])
            
            # Update P and X
            P_mask &= ~lsb
            X_mask |= lsb
            
            # Time check
            if time.time() > deadline:
                return

            temp_C ^= lsb

    # Initial call
    # R = 0
    # P = all nodes
    # X = 0
    bk(0, all_nodes_mask, 0, 0.0)
    
    # Convert best_set_mask to list of values
    result_indices = mask_to_list(best_set_mask)
    result = [values[i] for i in result_indices]
    
    if not result:
        return [2]
        
    # Verify
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if lcm(result[i], result[j]) <= 256:
                 valid = False
                 break
        if not valid:
            break
            
    if not valid:
        # Fallback to greedy
        greedy_indices = mask_to_list(R_greedy_mask)
        return [values[i] for i in greedy_indices]
        
    return result
