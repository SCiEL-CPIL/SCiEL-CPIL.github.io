"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import lcm
    from fractions import Fraction

    start_time = time.time()
    deadline = start_time + 85.0

    # Nodes are integers 2..256
    nodes = list(range(2, 257))
    N = len(nodes)
    
    # Weights are 1/x
    weights = [1.0 / x for x in nodes]
    
    # Build conflict masks
    # masks[i] is an integer where bit j is set if nodes[i] conflicts with nodes[j]
    masks = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            if lcm(nodes[i], nodes[j]) <= 256:
                masks[i] |= (1 << j)
                masks[j] |= (1 << i)

    def get_value(mask):
        val = 0.0
        for i in range(N):
            if (mask >> i) & 1:
                val += weights[i]
        return val

    def greedy_fill(seed_mask):
        cur_mask = seed_mask
        # Iterate through all nodes sorted by weight descending (node ascending)
        for i in range(N):
            if not ((cur_mask >> i) & 1):
                if (masks[i] & cur_mask) == 0:
                    cur_mask |= (1 << i)
        return cur_mask

    def local_search(init_mask):
        cur_mask = init_mask
        cur_val = get_value(cur_mask)
        best_mask = cur_mask
        best_val = cur_val
        
        # List of indices in current mask for faster iteration
        cur_indices = [i for i in range(N) if (cur_mask >> i) & 1]
        
        improved = True
        while improved:
            improved = False
            best_swap_gain = 0
            best_swap_u = -1
            best_swap_v = -1
            
            # Try single swaps
            for u in cur_indices:
                mask_without_u = cur_mask ^ (1 << u)
                # Iterate through all v not in cur_mask
                # Optimization: only check v that are compatible with mask_without_u
                # We can just iterate all v
                for v in range(N):
                    if not ((cur_mask >> v) & 1):
                        if (masks[v] & mask_without_u) == 0:
                            gain = weights[v] - weights[u]
                            if gain > best_swap_gain:
                                best_swap_gain = gain
                                best_swap_u = u
                                best_swap_v = v
            
            if best_swap_u != -1:
                # Apply swap
                cur_mask ^= (1 << best_swap_u)
                cur_mask |= (1 << best_swap_v)
                cur_val += best_swap_gain
                # Update indices
                cur_indices.remove(best_swap_u)
                cur_indices.append(best_swap_v)
                
                if cur_val > best_val:
                    best_val = cur_val
                    best_mask = cur_mask
                    
                improved = True
                
                # Time check inside loop
                if time.time() > deadline - 5.0:
                    break
            else:
                # No single swap improves. 
                # We could try 2-swaps, but let's stop for this restart to save time for other seeds
                break
                
        return best_mask, best_val

    best_result_mask = 0
    best_result_val = 0.0
    
    # Try different seeds
    # Seeds: small numbers 2..20, and maybe some random ones?
    # Small numbers are most likely to be in the optimal solution due to high weight.
    seeds = list(range(2, 21))
    
    for seed in seeds:
        if time.time() > deadline - 10.0:
            break
            
        seed_idx = seed - 2 # Index in nodes
        seed_mask = 1 << seed_idx
        
        # Greedy fill
        init_mask = greedy_fill(seed_mask)
        
        # Local search
        final_mask, val = local_search(init_mask)
        
        if val > best_result_val:
            best_result_val = val
            best_result_mask = final_mask

    # If no seed worked (shouldn't happen), fallback to empty? No, must be non-empty.
    # Fallback to greedy with no seed (just greedy)
    if best_result_mask == 0:
        init_mask = greedy_fill(0)
        final_mask, val = local_search(init_mask)
        best_result_mask = final_mask
        best_result_val = val

    # Extract result
    result = [nodes[i] for i in range(N) if (best_result_mask >> i) & 1]
    
    return result
# EVOLVE-BLOCK-END
