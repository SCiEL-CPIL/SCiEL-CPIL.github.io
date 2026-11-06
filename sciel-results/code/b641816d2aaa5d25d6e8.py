import sys

def construct():
    N = 256
    # 1. Build conflict graph (edges where divisibility holds)
    # We want to AVOID these edges. So we want Independent Set in this graph.
    # Which is Max Clique in the Complement Graph.
    # Let's build the "Non-Conflict" graph (Complement) directly for Max Clique.
    # non_adj[i] will have bits set for j where (i+1)*(j+1) % (i+1+j+1) != 0
    non_adj = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) != 0:
                non_adj[i] |= (1 << j)
                non_adj[j] |= (1 << i)
                
    # 2. Reorder vertices by descending degree to improve greedy coloring bound
    degrees = [non_adj[i].bit_count() for i in range(N)]
    order = sorted(range(N), key=lambda x: degrees[x], reverse=True)
    
    # Map old indices to new indices
    # new_index[old_index] = new_position
    new_index = [0] * N
    for new_i, old_i in enumerate(order):
        new_index[old_i] = new_i
        
    # Rebuild non_adj in new order
    new_non_adj = [0] * N
    for i in range(N):
        old_i = order[i]
        mask = non_adj[old_i]
        # Remap bits
        new_mask = 0
        m = mask
        while m:
            bit = m & -m
            old_j = bit.bit_length() - 1
            new_j = new_index[old_j]
            new_mask |= (1 << new_j)
            m &= m - 1
        new_non_adj[i] = new_mask

    # 3. Max Clique using Branch and Bound (Tomita style)
    best_size = 0
    best_set = 0
    
    def popcount(x):
        return x.bit_count()
        
    # Greedy coloring of the subgraph induced by 'mask' in the NON_ADJ graph
    # This gives an upper bound on the size of the Max Clique in that subgraph.
    def greedy_color_bound(mask):
        if mask == 0:
            return 0
        
        colors = 0
        remaining = mask
        while remaining:
            colors += 1
            # Find a maximal clique in the subgraph induced by 'remaining' in G (non_adj)
            # This clique will be assigned the current color in the complement graph coloring.
            current_clique = 0
            rem_for_clique = remaining
            while rem_for_clique:
                # Pick vertex with lowest index in 'rem_for_clique'.
                # Since we ordered by degree desc, this picks highest degree among remaining.
                v_bit = rem_for_clique & -rem_for_clique
                v_idx = v_bit.bit_length() - 1
                current_clique |= v_bit
                # Keep only neighbors of v in G
                rem_for_clique &= new_non_adj[v_idx]
                rem_for_clique &= ~v_bit
                
            remaining &= ~current_clique
            
        return colors

    def branch(candidates, selected, size):
        nonlocal best_size, best_set
        
        if candidates == 0:
            if size > best_size:
                best_size = size
                best_set = selected
            return
            
        # Upper bound
        if size + popcount(candidates) <= best_size:
            return
            
        # Better bound
        ub = size + greedy_color_bound(candidates)
        if ub <= best_size:
            return
            
        # Choose a vertex to branch on
        # Heuristic: Pick vertex with highest degree in the induced subgraph of 'candidates' in G
        # Since we ordered by global degree, the lowest index in 'candidates' is a good proxy.
        # But local degree is better. Let's stick to lowest index for speed, 
        # as the global ordering already biases towards high-degree nodes.
        v_bit = candidates & -candidates
        v_idx = v_bit.bit_length() - 1
        
        # Branch 1: Include v_idx
        new_candidates_include = candidates & new_non_adj[v_idx]
        branch(new_candidates_include, selected | v_bit, size + 1)
        
        # Branch 2: Exclude v_idx
        new_candidates_exclude = candidates & ~v_bit
        branch(new_candidates_exclude, selected, size)

    all_nodes = (1 << N) - 1
    branch(all_nodes, 0, 0)
    
    # Map best_set back to original indices
    result = []
    # best_set has bits set for new indices
    # We need to find which old indices correspond to these new indices
    # order[new_idx] = old_idx
    for new_idx in range(N):
        if (best_set >> new_idx) & 1:
            old_idx = order[new_idx]
            result.append(old_idx + 1)
            
    return result
