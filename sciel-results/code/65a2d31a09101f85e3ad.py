"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import math
    from itertools import combinations
    import time
    
    n = 128
    
    # 1. Precompute LCMs and find forbidden triples
    def lcm(a, b):
        return a * b // math.gcd(a, b)
        
    lcm_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            lcm_table[i][j] = lcm(i+1, j+1)
            lcm_table[j][i] = lcm_table[i][j]
            
    forbidden_triples = []
    for a in range(n):
        for b in range(a+1, n):
            lab = lcm_table[a][b]
            for c in range(b+1, n):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    forbidden_triples.append((a, b, c))
                    
    num_edges = len(forbidden_triples)
    
    if num_edges == 0:
        return list(range(1, n+1))
        
    # 2. Model as Minimum Hitting Set
    # We want to find the smallest set X of vertices such that every forbidden triple contains at least one vertex from X.
    # The solution is then [1, n] \ X.
    
    # Represent hyperedges as bitmasks (integers)
    edge_masks = []
    for tri in forbidden_triples:
        mask = (1 << tri[0]) | (1 << tri[1]) | (1 << tri[2])
        edge_masks.append(mask)
        
    # Precompute for each vertex v, the bitmask of edges containing v
    # This allows us to quickly determine which edges are "hit" by selecting v.
    # edges_containing[v] is a bitmask where bit i is 1 if edge i contains vertex v.
    edges_containing = [0] * n
    for i, mask in enumerate(edge_masks):
        # Iterate through set bits in mask
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            edges_containing[v] |= (1 << i)
            m ^= lsb
            
    # All edges mask
    all_edges_mask = (1 << num_edges) - 1
    
    # Lower Bound Heuristic: Greedy Disjoint Pack
    # Returns a lower bound on the number of additional vertices needed to hit all edges in 'active_edges'.
    def lower_bound(active_edges):
        count = 0
        remaining = active_edges
        while remaining:
            # Pick the first active edge
            lsb_edge = remaining & -remaining
            edge_idx = lsb_edge.bit_length() - 1
            edge_mask = edge_masks[edge_idx]
            
            count += 1
            
            # Remove all edges that intersect with this edge
            # An edge intersects if they share a vertex.
            # We can compute the union of edges_containing[v] for v in edge_mask.
            hit_edges = 0
            m = edge_mask
            while m:
                lsb_v = m & -m
                v = lsb_v.bit_length() - 1
                hit_edges |= edges_containing[v]
                m ^= lsb_v
                
            remaining &= ~hit_edges
            
        return count
        
    best_size = n # Worst case: exclude all
    best_hitting_set = (1 << n) - 1 # Bitmask of excluded vertices
    
    start_time = time.time()
    timeout = 55 # Seconds
    
    # Backtracking for Minimum Hitting Set
    # State: active_edges (bitmask), current_hitting_set (bitmask), current_size
    def solve(active_edges, current_hs, current_size):
        nonlocal best_size, best_hitting_set
        
        if time.time() - start_time > timeout:
            return
            
        if active_edges == 0:
            if current_size < best_size:
                best_size = current_size
                best_hitting_set = current_hs
            return
            
        # Pruning: If current_size + lower_bound(active_edges) >= best_size, we can't improve
        if current_size + lower_bound(active_edges) >= best_size:
            return
            
        # Select an edge to branch on.
        # Heuristic: Pick the edge with the fewest options? Or just the first one?
        # Since all edges have size 3, the branching factor is always 3.
        # We can pick the edge that appears in the most other edges (most constrained)?
        # For now, pick the first one for simplicity, or randomize? 
        # Picking the first one is deterministic.
        
        lsb_edge = active_edges & -active_edges
        edge_idx = lsb_edge.bit_length() - 1
        edge_mask = edge_masks[edge_idx]
        
        # The vertices in this edge
        vertices_in_edge = []
        m = edge_mask
        while m:
            lsb_v = m & -m
            v = lsb_v.bit_length() - 1
            vertices_in_edge.append(v)
            m ^= lsb_v
            
        # Branch on each vertex in the edge
        # We must select at least one of these vertices into the hitting set.
        for v in vertices_in_edge:
            new_hs = current_hs | (1 << v)
            new_size = current_size + 1
            
            # If we select v, all edges containing v are hit (removed from active)
            new_active = active_edges & ~edges_containing[v]
            
            solve(new_active, new_hs, new_size)
            
            # If we found a solution that is size 1 (impossible here as edges size 3) 
            # or if best_size is 1, we can stop. 
            # But we are minimizing, so if best_size == 1, we can stop.
            if best_size == 1:
                return

    # Initialize
    solve(all_edges_mask, 0, 0)
    
    # The result is the complement of best_hitting_set
    result_mask = ((1 << n) - 1) & ~best_hitting_set
    
    result = []
    for i in range(n):
        if result_mask & (1 << i):
            result.append(i + 1)
            
    return result
# EVOLVE-BLOCK-END
