# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    N = 256
    nodes = list(range(2, 257))
    num_nodes = len(nodes)
    # Map value to index 0..254
    val_to_idx = {v: i for i, v in enumerate(nodes)}
    idx_to_val = nodes
    
    # Precompute conflict masks
    # conflict_mask[i] is an integer bitmask where bit j is set if node i and node j conflict
    conflict_mask = [0] * num_nodes
    
    for i in range(num_nodes):
        vi = idx_to_val[i]
        mask = 0
        for j in range(num_nodes):
            if i == j:
                continue
            vj = idx_to_val[j]
            if math.lcm(vi, vj) <= 256:
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    def get_score(mask):
        score = 0.0
        # Iterate bits set in mask
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            score += 1.0 / idx_to_val[idx]
            m ^= lsb
        return score

    def local_search(initial_mask):
        current_mask = initial_mask
        current_score = get_score(current_mask)
        improved = True
        while improved:
            improved = False
            # Add phase: try to add any v not in mask
            # We prioritize adding smaller v (higher weight)
            # Iterate v in ascending order of value (2, 3, ...)
            for v in range(2, 257):
                if time.time() > deadline:
                    return current_mask, current_score
                idx = val_to_idx[v]
                if not (current_mask & (1 << idx)):
                    # Check conflict with current_mask
                    if (conflict_mask[idx] & current_mask) == 0:
                        current_mask |= (1 << idx)
                        current_score += 1.0 / v
                        improved = True
            
            # Swap phase: try to swap u in mask with v not in mask
            # We want 1/v > 1/u => v < u
            # Iterate u in descending order of value (larger u first, lower weight)
            # Iterate v in ascending order of value (smaller v first, higher weight)
            # To be efficient, we can just check all pairs where v < u
            # But we can prune: if v >= u, swap doesn't improve score.
            
            # Get list of indices in mask and not in mask
            # This is expensive if done every time. 
            # But num_nodes is 255.
            in_mask = []
            not_in_mask = []
            for i in range(num_nodes):
                if current_mask & (1 << i):
                    in_mask.append(i)
                else:
                    not_in_mask.append(i)
            
            # Sort in_mask by value descending (u large first)
            # Sort not_in_mask by value ascending (v small first)
            in_mask.sort(key=lambda i: idx_to_val[i], reverse=True)
            not_in_mask.sort(key=lambda i: idx_to_val[i])
            
            for u_idx in in_mask:
                if time.time() > deadline:
                    return current_mask, current_score
                u_val = idx_to_val[u_idx]
                for v_idx in not_in_mask:
                    v_val = idx_to_val[v_idx]
                    if v_val >= u_val:
                        continue # No score improvement
                    # Check if v is compatible with mask excluding u
                    mask_without_u = current_mask & ~(1 << u_idx)
                    if (conflict_mask[v_idx] & mask_without_u) == 0:
                        # Perform swap
                        current_mask = (current_mask & ~(1 << u_idx)) | (1 << v_idx)
                        current_score += 1.0 / v_val - 1.0 / u_val
                        improved = True
                        # Restart swap loop? Or continue?
                        # Restarting ensures we catch cascading improvements.
                        # But for speed, we can just break and let outer loop handle it.
                        break
                if improved:
                    break
        
        return current_mask, current_score

    # Strategy 1: Greedy by weight (small to large)
    mask1 = 0
    for v in range(2, 257):
        idx = val_to_idx[v]
        if (conflict_mask[idx] & mask1) == 0:
            mask1 |= (1 << idx)
    
    # Strategy 2: Large numbers set {129..256}
    mask2 = 0
    for v in range(129, 257):
        idx = val_to_idx[v]
        mask2 |= (1 << idx)
        
    # Strategy 3: Greedy by "least conflicts" among high weight?
    # Let's just run local search on both starts.
    
    best_mask = 0
    best_score = 0.0
    
    m1, s1 = local_search(mask1)
    if s1 > best_score:
        best_score = s1
        best_mask = m1
        
    if time.time() < deadline:
        m2, s2 = local_search(mask2)
        if s2 > best_score:
            best_score = s2
            best_mask = m2
            
    # Convert mask to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx_to_val[idx])
        m ^= lsb
        
    return result
# EVOLVE-BLOCK-END
