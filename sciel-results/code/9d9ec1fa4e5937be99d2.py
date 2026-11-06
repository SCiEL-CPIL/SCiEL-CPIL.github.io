import time

def construct():
    start_time = time.time()
    timeout = 55.0
    
    n = 256
    
    # Build conflict masks
    # conflict[i] is a bitmask of all j such that (i+1) and (j+1) conflict
    conflict = [0] * n
    
    for i in range(n):
        a = i + 1
        mask_i = 0
        for j in range(n):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                mask_i |= (1 << j)
        conflict[i] = mask_i
    
    best_size = 0
    best_mask = 0
    
    def popcount(x):
        return x.bit_count()
    
    # Precompute degrees for ordering
    degrees = [conflict[i].bit_count() for i in range(n)]
    
    # Vertex ordering: sort by degree descending
    order = sorted(range(n), key=lambda i: degrees[i], reverse=True)
    # Map original index to position in order
    pos_in_order = [0] * n
    for idx, orig in enumerate(order):
        pos_in_order[orig] = idx
    
    # Reorder conflict masks to match the new ordering
    # new_conflict[i] corresponds to original vertex order[i]
    new_conflict = [0] * n
    for i in range(n):
        orig_i = order[i]
        mask = 0
        for j in range(n):
            if conflict[orig_i] & (1 << j):
                mask |= (1 << pos_in_order[j])
        new_conflict[i] = mask
    
    def solve(candidates, current_size, current_mask):
        nonlocal best_size, best_mask
        
        if time.time() - start_time > timeout:
            return
            
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
            
        # Pruning
        if current_size + popcount(candidates) <= best_size:
            return
            
        # Select vertex v from candidates
        # Use max degree heuristic in the induced subgraph
        v_idx = -1
        max_deg = -1
        c = candidates
        while c:
            lsb = c & -c
            i = lsb.bit_length() - 1
            deg = popcount(new_conflict[i] & candidates)
            if deg > max_deg:
                max_deg = deg
                v_idx = i
            c ^= lsb
            
        if v_idx == -1:
            return
            
        v_bit = 1 << v_idx
        
        # Branch 1: Include v_idx
        new_cand_inc = (candidates & ~v_bit) & ~new_conflict[v_idx]
        new_mask_inc = current_mask | v_bit
        solve(new_cand_inc, current_size + 1, new_mask_inc)
        
        # Branch 2: Exclude v_idx
        new_cand_exc = candidates & ~v_bit
        solve(new_cand_exc, current_size, current_mask)
        
    # Initial call
    initial_cand = (1 << n) - 1
    solve(initial_cand, 0, 0)
    
    if best_size == 0:
        return [1]
        
    # Convert best_mask back to list of integers
    # The mask is in the reordered space, so we need to map back
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(order[i] + 1)
            
    return result
