import random

def construct():
    n = 20
    best_edge_list = []
    max_edges = 0
    
    def is_valid_addition(edge_colors, adj, u, v, c):
        # Check if adding edge (u, v) with color c creates a non-rainbow 4-cycle
        # A 4-cycle is u-a-b-v-u
        # a must be in adj[u]
        # b must be in adj[v]
        # a and b must be connected
        # a, b, u, v must be distinct
        
        for a in adj[u]:
            if a == v:
                continue
            # a is a neighbor of u
            c_ua = edge_colors.get((min(u, a), max(u, a)))
            if c_ua is None:
                continue # Should not happen if adj is consistent
            
            for b in adj[v]:
                if b == u or b == a:
                    continue
                # b is a neighbor of v, distinct from u and a
                
                # Check if a and b are connected
                key_ab = (min(a, b), max(a, b))
                c_ab = edge_colors.get(key_ab)
                if c_ab is None:
                    continue
                
                c_bv = edge_colors.get((min(b, v), max(b, v)))
                # c_bv must exist because b in adj[v]
                
                # Check distinctness of {c, c_ua, c_ab, c_bv}
                if len({c, c_ua, c_ab, c_bv}) < 4:
                    return False
        
        return True

    def add_edge(edge_colors, adj, u, v, c):
        key = (min(u, v), max(u, v))
        edge_colors[key] = c
        adj[u].add(v)
        adj[v].add(u)

    # Strategy:
    # 1. Initialize with K(10,10) using modular coloring c(u,v) = (u+v) % 20
    # 2. Greedily add remaining edges (internal to parts)
    
    edge_colors = {}
    adj = [set() for _ in range(n)]
    
    # Part A: 0-9, Part B: 10-19
    part_a = list(range(10))
    part_b = list(range(10, 20))
    
    # Add bipartite edges
    for u in part_a:
        for v in part_b:
            c = (u + v) % 20
            # This coloring is proven to be rainbow for 4-cycles within K(10,10)
            # We add them directly
            key = (u, v)
            edge_colors[key] = c
            adj[u].add(v)
            adj[v].add(u)
            
    # Now try to add internal edges
    # Internal edges in A
    internal_edges_a = [(u, v) for u in part_a for v in part_a if u < v]
    # Internal edges in B
    internal_edges_b = [(u, v) for u in part_b for v in part_b if u < v]
    
    all_remaining = internal_edges_a + internal_edges_b
    
    # Shuffle to handle conflicts better? Or just iterate.
    # Randomizing might help avoid local minima if we allow skipping.
    random.seed(42)
    random.shuffle(all_remaining)
    
    for u, v in all_remaining:
        found_color = None
        # Try colors 0..19
        for c in range(20):
            if is_valid_addition(edge_colors, adj, u, v, c):
                found_color = c
                break
        
        if found_color is not None:
            add_edge(edge_colors, adj, u, v, found_color)
            
    current_edges = len(edge_colors)
    
    # If this specific construction yields fewer edges than a generic greedy on K(20), 
    # we might want to compare. But K(10,10) + internal is likely very dense.
    # Let's also run a few generic greedy attempts on K(20) to see if they beat this.
    
    def try_greedy_full(order):
        edge_colors_g = {}
        adj_g = [set() for _ in range(n)]
        
        for u, v in order:
            key = (min(u, v), max(u, v))
            if key in edge_colors_g:
                continue
            
            found_color = None
            for c in range(20):
                if is_valid_addition(edge_colors_g, adj_g, u, v, c):
                    found_color = c
                    break
            
            if found_color is not None:
                add_edge(edge_colors_g, adj_g, u, v, found_color)
                
        return len(edge_colors_g), edge_colors_g

    all_edges_k20 = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    for attempt in range(20):
        random.seed(attempt)
        random.shuffle(all_edges_k20)
        num_edges, edge_dict = try_greedy_full(all_edges_k20)
        
        if num_edges > current_edges:
            current_edges = num_edges
            edge_colors = edge_dict
            adj = [set() for _ in range(n)]
            for (u, v), c in edge_dict.items():
                adj[u].add(v)
                adj[v].add(u)
            
    best_edge_list = [[u, v, c] for (u, v), c in edge_colors.items()]
    return best_edge_list
