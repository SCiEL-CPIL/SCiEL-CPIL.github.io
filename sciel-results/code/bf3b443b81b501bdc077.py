# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit
    
    N = 256
    nodes = list(range(2, 257))
    num_nodes = len(nodes)
    val_to_idx = {v: i for i, v in enumerate(nodes)}
    idx_to_val = nodes
    
    # Precompute conflict masks
    # conflict_mask[i] is an integer bitmask where bit j is set if node i and node j conflict (lcm <= 256)
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
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            score += 1.0 / idx_to_val[idx]
            m ^= lsb
        return score

    def local_search(initial_mask, time_limit=5.0):
        current_mask = initial_mask
        current_score = get_score(current_mask)
        improved = True
        start_ls_time = time.time()
        
        while improved:
            improved = False
            
            # Check time
            if time.time() > deadline or time.time() - start_ls_time > time_limit:
                break
            
            # Add phase: try to add any v not in mask
            # Prioritize adding smaller v (higher weight)
            for v in range(2, 257):
                idx = val_to_idx[v]
                if not (current_mask & (1 << idx)):
                    if (conflict_mask[idx] & current_mask) == 0:
                        current_mask |= (1 << idx)
                        current_score += 1.0 / v
                        improved = True
            
            # Swap phase: try to swap u in mask with v not in mask
            # We want 1/v > 1/u => v < u
            # Iterate u in descending order of value (larger u first, lower weight)
            # Iterate v in ascending order of value (smaller v first, higher weight)
            
            in_mask = []
            not_in_mask = []
            for i in range(num_nodes):
                if current_mask & (1 << i):
                    in_mask.append(i)
                else:
                    not_in_mask.append(i)
            
            in_mask.sort(key=lambda i: idx_to_val[i], reverse=True)
            not_in_mask.sort(key=lambda i: idx_to_val[i])
            
            # Optimization: Precompute list of (v_idx, v_val) for not_in_mask
            # and iterate.
            found_swap = False
            for u_idx in in_mask:
                if time.time() > deadline:
                    return current_mask, current_score
                u_val = idx_to_val[u_idx]
                mask_without_u = current_mask & ~(1 << u_idx)
                
                for v_idx in not_in_mask:
                    v_val = idx_to_val[v_idx]
                    if v_val >= u_val:
                        continue # No score improvement
                    
                    # Check if v is compatible with mask excluding u
                    if (conflict_mask[v_idx] & mask_without_u) == 0:
                        # Perform swap
                        current_mask = (current_mask & ~(1 << u_idx)) | (1 << v_idx)
                        current_score += 1.0 / v_val - 1.0 / u_val
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    # Restart swap loop to catch cascading improvements
                    # But to save time, we can just break and let the outer while loop handle it
                    break
        
        return current_mask, current_score

    def perturb(mask):
        # Remove a random element from the mask
        if mask == 0:
            return mask
        bits = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            bits.append(idx)
            m ^= lsb
        
        if not bits:
            return mask
            
        # Remove 1 or 2 random elements to create a gap
        num_remove = random.choice([1, 1, 2])
        indices_to_remove = random.sample(bits, min(num_remove, len(bits)))
        
        new_mask = mask
        for idx in indices_to_remove:
            new_mask &= ~(1 << idx)
        return new_mask

    best_mask = 0
    best_score = 0.0

    # Generate diverse initial solutions
    # 1. Greedy by weight (small to large)
    mask1 = 0
    for v in range(2, 257):
        idx = val_to_idx[v]
        if (conflict_mask[idx] & mask1) == 0:
            mask1 |= (1 << idx)
    
    # 2. Greedy by "least conflicts" 
    # Calculate initial conflict count for each node
    # Actually, let's just use a randomized greedy: 
    # At each step, pick the available node with highest weight (smallest value) 
    # from a random subset of top candidates? 
    # Simpler: Randomized Greedy
    # Shuffle nodes, but bias towards smaller values?
    # Let's try: Pick random node, if it fits, add it. Repeat until no more fit.
    # To make it better, pick from candidates sorted by weight with some noise.
    
    def randomized_greedy():
        m = 0
        available = list(range(2, 257))
        while available:
            # Filter available to those compatible with current mask
            compatible = [v for v in available if (conflict_mask[val_to_idx[v]] & m) == 0]
            if not compatible:
                break
            # Pick one from compatible. 
            # Bias towards smaller values (higher weight).
            # Sort compatible by value, pick from top 10% or random?
            compatible.sort()
            # Pick randomly from the first few smallest (highest weight)
            k = min(5, len(compatible))
            v = random.choice(compatible[:k])
            idx = val_to_idx[v]
            m |= (1 << idx)
            available.remove(v)
        return m

    # 3. Large numbers set {129..256}
    mask2 = 0
    for v in range(129, 257):
        idx = val_to_idx[v]
        mask2 |= (1 << idx)
        
    # 4. Another randomized greedy
    mask3 = randomized_greedy()
    
    initial_masks = [mask1, mask2, mask3]
    
    # Add more randomized starts if time permits
    num_restarts = 0
    max_restarts = 20
    
    for i in range(max_restarts):
        if time.time() > deadline - 10.0: # Reserve time for final LS
            break
            
        # Generate a new initial mask via randomized greedy
        m_init = randomized_greedy()
        
        # Local search
        m_ls, s_ls = local_search(m_init, time_limit=2.0)
        
        if s_ls > best_score:
            best_score = s_ls
            best_mask = m_ls
            
        num_restarts += 1
        
        # If we found a good solution, we can try to perturb it
        if num_restarts % 3 == 0 and best_mask != 0:
             m_pert = perturb(best_mask)
             m_ls2, s_ls2 = local_search(m_pert, time_limit=2.0)
             if s_ls2 > best_score:
                 best_score = s_ls2
                 best_mask = m_ls2

    # Ensure we run LS on the initial specific masks if not done
    if mask1 != best_mask:
        m1, s1 = local_search(mask1, time_limit=5.0)
        if s1 > best_score:
            best_score = s1
            best_mask = m1
            
    if mask2 != best_mask:
        m2, s2 = local_search(mask2, time_limit=5.0)
        if s2 > best_score:
            best_score = s2
            best_mask = m2

    if mask3 != best_mask:
        m3, s3 = local_search(mask3, time_limit=5.0)
        if s3 > best_score:
            best_score = s3
            best_mask = m3

    # Final polish on the best found
    if time.time() < deadline - 2.0:
        m_final, s_final = local_search(best_mask, time_limit=deadline - time.time() - 1.0)
        if s_final > best_score:
            best_score = s_final
            best_mask = m_final

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
