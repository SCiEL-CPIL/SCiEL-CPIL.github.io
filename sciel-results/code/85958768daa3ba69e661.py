def construct():
    import time
    start_time = time.time()
    DEADLINE = 85.0
    
    from math import lcm
    
    num_vertices = 255
    values = list(range(2, 257))
    weights = [1.0 / v for v in values]
    
    # Build conflict masks
    conflict_mask = [0] * num_vertices
    for i in range(num_vertices):
        a = values[i]
        for j in range(i + 1, num_vertices):
            b = values[j]
            if lcm(a, b) <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Precompute chunk sums
    num_chunks = (num_vertices + 15) // 16
    chunk_sums = []
    for c in range(num_chunks):
        table = [0.0] * (1 << 16)
        for mask in range(1, 1 << 16):
            lowbit = mask & (-mask)
            bit = lowbit.bit_length() - 1
            rest = mask ^ lowbit
            idx = c * 16 + bit
            if idx < num_vertices:
                table[mask] = table[rest] + weights[idx]
            else:
                table[mask] = table[rest]
        chunk_sums.append(table)
    
    def sum_weights(mask):
        s = 0.0
        for c in range(num_chunks):
            chunk = (mask >> (c * 16)) & 0xFFFF
            s += chunk_sums[c][chunk]
        return s
    
    all_mask = (1 << num_vertices) - 1
    
    # Greedy initialization (multiple orders)
    best_score = 0.0
    best_set = []
    
    # Greedy in natural order (highest weight first)
    greedy_mask = all_mask
    greedy_set = []
    for i in range(num_vertices):
        if greedy_mask & (1 << i):
            greedy_set.append(i)
            greedy_mask &= ~(conflict_mask[i] | (1 << i))
    gs = sum(weights[i] for i in greedy_set)
    if gs > best_score:
        best_score = gs
        best_set = list(greedy_set)
    
    import sys
    sys.setrecursionlimit(10000)
    
    def bb(candidates, current_sum, current_set):
        nonlocal best_score, best_set
        
        if time.time() - start_time > DEADLINE:
            return
        
        if candidates == 0:
            if current_sum > best_score:
                best_score = current_sum
                best_set = list(current_set)
            return
        
        ub = current_sum + sum_weights(candidates)
        if ub <= best_score:
            return
        
        # Pick branch vertex
        m = candidates
        branch_idx = (m & (-m)).bit_length() - 1
        
        # Branch 1: include
        new_cand = candidates & ~(conflict_mask[branch_idx] | (1 << branch_idx))
        new_sum = current_sum + weights[branch_idx]
        if new_sum + sum_weights(new_cand) > best_score:
            bb(new_cand, new_sum, current_set + [branch_idx])
        
        # Branch 2: exclude
        new_cand2 = candidates & ~(1 << branch_idx)
        if current_sum + sum_weights(new_cand2) > best_score:
            bb(new_cand2, current_sum, current_set)
    
    bb(all_mask, 0.0, [])
    
    return sorted([i + 2 for i in best_set])
