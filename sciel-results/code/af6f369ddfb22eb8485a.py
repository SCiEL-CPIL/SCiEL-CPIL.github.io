import time

def construct():
    start_time = time.time()
    timeout = 55.0
    
    n = 256
    
    # Build conflict masks
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
    
    # Precompute degrees
    degrees = [conflict[i].bit_count() for i in range(n)]
    
    best_size = 0
    best_mask = 0
    
    def popcount(x):
        return x.bit_count()
    
    # Greedy coloring upper bound on the induced subgraph of candidates
    # This gives an upper bound on the maximum independent set size
    def color_bound(candidates):
        """Return upper bound on MIS size in the induced subgraph of candidates."""
        if candidates == 0:
            return 0
        
        # Greedy sequential coloring of the complement graph (conflict graph)
        # We want to color the conflict graph so that adjacent vertices have different colors.
        # The number of colors needed is an upper bound on the independence number... 
        # Wait, no. For MIS, we need a lower bound on the clique cover of the complement,
        # or equivalently, an upper bound via coloring the conflict graph.
        # Actually: alpha(G) <= chi(G) (chromatic number of G)
        # So coloring the conflict graph gives an upper bound on MIS size.
        
        # Greedy coloring: assign each vertex the smallest available color
        uncolored = candidates
        colors = {}
        color_count = 0
        
        while uncolored:
            # Pick vertex with max degree in uncolored subgraph
            v_idx = -1
            max_deg = -1
            c = uncolored
            while c:
                lsb = c & -c
                i = lsb.bit_length() - 1
                deg = (conflict[i] & uncolored).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v_idx = i
                c ^= lsb
            
            if v_idx == -1:
                break
            
            # Find smallest color not used by colored neighbors
            used_colors = set()
            neighbors_colored = conflict[v_idx] & ~uncolored
            nc = neighbors_colored
            while nc:
                lsb = nc & -nc
                j = lsb.bit_length() - 1
                if j in colors:
                    used_colors.add(colors[j])
                nc ^= lsb
            
            for col in range(color_count + 1):
                if col not in used_colors:
                    colors[v_idx] = col
                    if col == color_count:
                        color_count += 1
                    break
            
            uncolored &= ~(1 << v_idx)
        
        return color_count
    
    def solve(candidates, current_size, current_mask, depth):
        nonlocal best_size, best_mask
        
        if time.time() - start_time > timeout:
            return
            
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
            
        # Pruning with greedy coloring bound
        if current_size + popcount(candidates) <= best_size:
            return
        
        # Use coloring bound for tighter pruning (only when candidate set is small enough)
        if popcount(candidates) <= 60:
            ub = color_bound(candidates)
            if current_size + ub <= best_size:
                return
        
        # Select vertex v from candidates using max-degree heuristic
        v_idx = -1
        max_deg = -1
        c = candidates
        while c:
            lsb = c & -c
            i = lsb.bit_length() - 1
            deg = (conflict[i] & candidates).bit_count()
            if deg > max_deg:
                max_deg = deg
                v_idx = i
            c ^= lsb
        
        if v_idx == -1:
            return
            
        v_bit = 1 << v_idx
        
        # Branch 1: Include v_idx
        new_cand_inc = (candidates & ~v_bit) & ~conflict[v_idx]
        new_mask_inc = current_mask | v_bit
        solve(new_cand_inc, current_size + 1, new_mask_inc, depth + 1)
        
        if time.time() - start_time > timeout:
            return
        
        # Branch 2: Exclude v_idx
        new_cand_exc = candidates & ~v_bit
        solve(new_cand_exc, current_size, current_mask, depth + 1)
    
    # Initial call
    initial_cand = (1 << n) - 1
    solve(initial_cand, 0, 0, 0)
    
    if best_size == 0:
        return [1]
        
    # Convert best_mask back to list of integers
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
