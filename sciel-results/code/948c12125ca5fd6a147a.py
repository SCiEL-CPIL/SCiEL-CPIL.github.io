import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    nodes = list(range(2, 257))
    n = len(nodes)
    # index 0 corresponds to 2, index 1 to 3, etc.
    # node_val[i] = i + 2
    
    weights = [1.0 / (i + 2) for i in range(n)]
    
    # Build conflict masks
    # conflict[i] is a bitmask where bit j is 1 if nodes[i] and nodes[j] conflict (lcm <= 256)
    conflict = [0] * n
    for i in range(n):
        a = i + 2
        mask = 0
        for j in range(n):
            if i == j:
                continue
            b = j + 2
            # Check lcm(a, b) <= 256
            # lcm(a, b) = a * b // gcd(a, b)
            if a * b // math.gcd(a, b) <= 256:
                mask |= (1 << j)
        conflict[i] = mask
        
    # Greedy initialization
    # Sort indices by weight descending
    indices = sorted(range(n), key=lambda i: weights[i], reverse=True)
    
    def greedy_init():
        selected = []
        sel_mask = 0
        for i in indices:
            # Check if i is compatible with all selected
            # Compatible means no conflict
            # conflict[i] has 1s for conflicts.
            # We need (sel_mask & conflict[i]) == 0
            if (sel_mask & conflict[i]) == 0:
                selected.append(i)
                sel_mask |= (1 << i)
        return selected, sum(weights[i] for i in selected)
        
    best_set, best_weight = greedy_init()
    
    # Branch and Bound
    # We will use a stack to avoid recursion depth issues, or recursion with sys.setrecursionlimit
    # Given n=255, recursion depth is at most 255. Python default is 1000, so it's fine.
    
    import sys
    sys.setrecursionlimit(3000)
    
    # Precompute sum of weights for quick bound?
    # sum_weights_mask(mask): sum of weights[i] for i where bit i is set in mask.
    # This is O(n) per call. n=255.
    # In B&B, this is called at every node.
    # To speed up, we can pass the current sum and subtract added weights.
    
    def search(candidates_mask, current_weight, current_set_mask, depth):
        nonlocal best_set, best_weight
        
        if time.time() > deadline:
            return
            
        if candidates_mask == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = current_set_mask
            return
            
        # Bound: current_weight + sum of all remaining weights
        # Calculate sum of weights in candidates_mask
        remaining_sum = 0.0
        temp_mask = candidates_mask
        while temp_mask:
            # Get lowest set bit
            lsb = temp_mask & -temp_mask
            idx = lsb.bit_length() - 1
            remaining_sum += weights[idx]
            temp_mask ^= lsb
            
        if current_weight + remaining_sum <= best_weight:
            return
            
        # Select a vertex to branch on
        # Heuristic: Pick vertex with highest weight in candidates?
        # Or highest degree in conflict graph within candidates?
        # High conflict degree means including it removes many options, but it has high weight?
        # Usually, for independent set, picking a vertex with high degree in the conflict graph
        # (i.e., many conflicts) is good for pruning because including it reduces the candidate set significantly.
        
        # Let's pick the vertex with the highest weight among those with the highest conflict degree in candidates.
        # Or simply highest weight.
        
        # To make it deterministic and fast, let's pick the vertex with the highest weight.
        # We can iterate through candidates.
        
        best_idx = -1
        max_w = -1.0
        temp_mask = candidates_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            idx = lsb.bit_length() - 1
            if weights[idx] > max_w:
                max_w = weights[idx]
                best_idx = idx
            temp_mask ^= lsb
            
        if best_idx == -1:
            return
            
        # Branch 1: Include best_idx
        # New candidates: candidates that are compatible with best_idx
        # Compatible means NOT in conflict[best_idx]
        # new_candidates = candidates_mask & ~conflict[best_idx]
        # Note: best_idx itself must be removed from candidates for the next step?
        # In the recursive call, we don't include best_idx again.
        # The standard formulation:
        # Include v: search(candidates \ {v} & N_compat(v), ... + w(v))
        # Exclude v: search(candidates \ {v}, ...)
        
        # Remove best_idx from candidates
        cand_without_v = candidates_mask & ~(1 << best_idx)
        
        # Include branch
        # Candidates must be compatible with best_idx
        # conflict[best_idx] has 1s for nodes that conflict.
        # We want nodes that do NOT conflict.
        # So we mask out conflict[best_idx]
        new_cands_inc = cand_without_v & ~conflict[best_idx]
        
        search(new_cands_inc, current_weight + weights[best_idx], current_set_mask | (1 << best_idx), depth + 1)
        
        # Exclude branch
        search(cand_without_v, current_weight, current_set_mask, depth + 1)
        
    # Start search
    # Initial candidates: all nodes
    initial_mask = (1 << n) - 1
    search(initial_mask, 0.0, 0, 0)
    
    # Reconstruct set from best_set mask
    result = []
    mask = best_set
    while mask:
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        result.append(idx + 2)
        mask ^= lsb
        
    return result
