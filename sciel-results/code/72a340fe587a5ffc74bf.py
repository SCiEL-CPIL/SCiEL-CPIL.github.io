import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    max_sum = 2 * N
    
    # Precompute squares up to max_sum
    squares = set()
    k = 2
    while k * k <= max_sum:
        squares.add(k * k)
        k += 1
    squares_list = sorted(squares)
    
    # Identify forbidden vertices (where 2*a is a perfect square)
    forbidden = set()
    for a in range(1, N + 1):
        s = 2 * a
        r = math.isqrt(s)
        if r * r == s:
            forbidden.add(a)
    
    # Compact indexing: only valid vertices
    valid_nodes = [i for i in range(1, N + 1) if i not in forbidden]
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    num_nodes = len(valid_nodes)
    
    # Build conflict masks using compact indices
    conflict_masks = [0] * num_nodes
    for i in range(num_nodes):
        a = valid_nodes[i]
        mask = 0
        for s in squares_list:
            b = s - a
            if 1 <= b <= N and b in node_to_idx:
                j = node_to_idx[b]
                if j != i:
                    mask |= (1 << j)
        conflict_masks[i] = mask
    
    # Initial candidates: all valid vertices
    all_candidates = (1 << num_nodes) - 1
    
    # Greedy MIS for initial solution
    def greedy_mis(cand_mask):
        selected = 0
        mask = cand_mask
        while mask:
            best_v = -1
            min_deg = num_nodes + 1
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = (conflict_masks[v] & mask).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
                m ^= lsb
            if best_v == -1:
                break
            selected |= (1 << best_v)
            mask &= ~(conflict_masks[best_v] | (1 << best_v))
        return selected
    
    best_set = greedy_mis(all_candidates)
    best_size = best_set.bit_count()
    
    # Fast coloring bound for upper bound on MIS size
    def coloring_bound(mask):
        if mask == 0:
            return 0
        uncolored = mask
        colors = 0
        while uncolored:
            colors += 1
            # Greedy maximal independent set for this color class
            m = uncolored
            while m:
                # Pick vertex with min degree in current pool
                best_v = -1
                min_deg = num_nodes + 1
                mm = m
                while mm:
                    lsb = mm & -mm
                    v = lsb.bit_length() - 1
                    deg = (conflict_masks[v] & m).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v
                    mm ^= lsb
                if best_v == -1:
                    break
                uncolored &= ~(conflict_masks[best_v] | (1 << best_v))
                m &= ~(conflict_masks[best_v] | (1 << best_v))
        return colors
    
    # Branch and Bound DFS
    def dfs(mask, current_size, current_set):
        nonlocal best_size, best_set
        if time.time() > deadline:
            return
        
        if mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
        
        # Upper bound: simple bit count
        ub_simple = mask.bit_count()
        if current_size + ub_simple <= best_size:
            return
        
        # Tighter bound: coloring
        ub_color = coloring_bound(mask)
        if current_size + ub_color <= best_size:
            return
        
        # Select vertex with max degree in mask for branching
        best_v = -1
        max_deg = -1
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            deg = (conflict_masks[v] & mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            m ^= lsb
        
        if best_v == -1:
            return
        
        # Branch 1: Include best_v
        new_mask_1 = mask & ~(conflict_masks[best_v] | (1 << best_v))
        dfs(new_mask_1, current_size + 1, current_set | (1 << best_v))
        
        # Branch 2: Exclude best_v
        new_mask_2 = mask & ~(1 << best_v)
        dfs(new_mask_2, current_size, current_set)
    
    dfs(all_candidates, 0, 0)
    
    # Convert bitmask to list of actual numbers
    result = []
    m = best_set
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(valid_nodes[idx])
        m ^= lsb
    
    return result
