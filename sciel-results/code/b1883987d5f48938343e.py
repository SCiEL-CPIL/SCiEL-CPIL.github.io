import random

def construct():
    n = 20
    best_edge_list = []
    max_edges = 0
    
    # We will run multiple greedy attempts
    # The key improvement is the efficient checking of 4-cycles
    # and potentially starting with a dense subgraph like K(10,10)
    
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

    def try_greedy(order):
        edge_colors = {}
        adj = [set() for _ in range(n)]
        
        for u, v in order:
            key = (min(u, v), max(u, v))
            if key in edge_colors:
                continue
            
            found_color = None
            # Try colors in a randomized order or fixed?
            # Fixed 0..19 is deterministic given the order.
            # Randomizing color order might help avoid local minima?
            # Let's try fixed first, as it's faster.
            for c in range(20):
                if is_valid_addition(edge_colors, adj, u, v, c):
                    found_color = c
                    break
            
            if found_color is not None:
                add_edge(edge_colors, adj, u, v, found_color)
                
        return len(edge_colors), edge_colors

    # Generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Run multiple attempts with different random orders
    for attempt in range(50):
        random.seed(attempt)
        random.shuffle(all_edges)
        
        # Optimization: We can try to prioritize edges in a bipartite structure first?
        # Or just random.
        
        num_edges, edge_dict = try_greedy(all_edges)
        
        if num_edges > max_edges:
            max_edges = num_edges
            best_edge_list = [[u, v, c] for (u, v), c in edge_dict.items()]
            
    # If we didn't get a very high number, try a specific K(10,10) construction
    # to ensure we have at least 100 edges if possible
    if max_edges < 100:
        # Construct K(10,10) greedily
        a_part = list(range(10))
        b_part = list(range(10, 20))
        bip_edges = [(u, v) for u in a_part for v in b_part]
        random.seed(123)
        random.shuffle(bip_edges)
        
        num_bip, bip_dict = try_greedy(bip_edges)
        if num_bip > max_edges:
            max_edges = num_bip
            best_edge_list = [[u, v, c] for (u, v), c in bip_dict.items()]

    return best_edge_list
