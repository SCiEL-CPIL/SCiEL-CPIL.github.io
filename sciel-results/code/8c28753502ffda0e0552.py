def construct():
    import time
    import random
    import numpy as np
    from itertools import combinations

    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    n = 20
    num_colors = 20

    # Precompute all 4-cycles as (u, v, w, x) where u-v-w-x-u is a cycle
    # u < v < w < x with edges (u,v), (v,w), (w,x), (x,u)
    # We need to be careful: a 4-cycle is any set of 4 vertices {a,b,c,d} 
    # with all 4 edges present in some cyclic order.
    # For vertices a < b < c < d, the possible 4-cycles using all 4 are:
    # a-b-c-d-a, a-b-d-c-a, a-c-b-d-a
    # But we only need to check when all 4 edges of the cycle exist.
    
    # Let's precompute all potential 4-cycles (as edge tuples)
    # A 4-cycle on 4 vertices {a,b,c,d} has 3 distinct cycles:
    # (a,b,c,d), (a,b,d,c), (a,c,b,d) - these are the 3 ways to form a cycle
    # Each cycle uses 4 edges.
    
    # For efficiency, let's precompute for each pair of edges, which 4-cycles they participate in
    # Actually, let's use a different approach: for each potential new edge (u,v),
    # we need to check all 4-cycles that would include (u,v).
    
    # A 4-cycle containing edge (u,v) has the form u-v-x-w-u where:
    # - w is adjacent to u (edge u-w exists)
    # - x is adjacent to v (edge v-x exists)  
    # - w and x are adjacent (edge w-x exists)
    # - w != v, x != u, w != x
    
    # Let's precompute the adjacency structure and use vectorized checks
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    edge_index = {}
    for i, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    def check_rainbow(adj_matrix, edge_u, edge_v, color):
        """Check if adding edge (edge_u, edge_v) with given color creates a non-rainbow 4-cycle."""
        u, v = edge_u, edge_v
        
        # Find all w such that (u,w) is an edge
        w_candidates = [w for w in range(n) if w != v and adj_matrix[u][w] != -1]
        # Find all x such that (v,x) is an edge
        x_candidates = [x for x in range(n) if x != u and adj_matrix[v][x] != -1]
        
        for w in w_candidates:
            for x in x_candidates:
                if w == x:
                    continue
                if adj_matrix[w][x] != -1:
                    # 4-cycle: u-v-x-w-u
                    c1 = color
                    c2 = adj_matrix[v][x]
                    c3 = adj_matrix[x][w]
                    c4 = adj_matrix[w][u]
                    if len({c1, c2, c3, c4}) != 4:
                        return False
        return True

    def try_greedy_construction(edge_order, color_order_fn):
        """Try to build a valid coloring by greedily adding edges."""
        adj_matrix = [[-1]*n for _ in range(n)]
        neighbors = [[] for _ in range(n)]
        current_edges = []
        
        for u, v in edge_order:
            # Try colors in the given order
            colors = color_order_fn()
            for c in colors:
                if check_rainbow(adj_matrix, u, v, c):
                    adj_matrix[u][v] = c
                    adj_matrix[v][u] = c
                    neighbors[u].append(v)
                    neighbors[v].append(u)
                    current_edges.append([u, v, c])
                    break
        
        return current_edges, adj_matrix, neighbors

    def try_remove_and_readd(adj_matrix, neighbors, current_edges):
        """Try to improve by removing some edges and re-adding them differently."""
        best_edges = current_edges[:]
        best_adj = [row[:] for row in adj_matrix]
        best_neighbors = [lst[:] for lst in neighbors]
        
        # Try removing each edge and see if we can add more
        for i in range(len(current_edges)):
            if time.time() > deadline - 5:
                break
            u, v, c = current_edges[i]
            
            # Temporarily remove
            adj_matrix[u][v] = -1
            adj_matrix[v][u] = -1
            neighbors[u].remove(v)
            neighbors[v].remove(u)
            remaining = current_edges[:i] + current_edges[i+1:]
            
            # Try to add back all missing edges
            new_adj = [row[:] for row in adj_matrix]
            new_neighbors = [lst[:] for lst in neighbors]
            new_edges = remaining[:]
            
            # First try to add the removed edge with a different color
            for c2 in range(num_colors):
                if check_rainbow(new_adj, u, v, c2):
                    new_adj[u][v] = c2
                    new_adj[v][u] = c2
                    new_neighbors[u].append(v)
                    new_neighbors[v].append(u)
                    new_edges.append([u, v, c2])
                    break
            
            # Try to add other missing edges
            for (a, b) in all_edges:
                if time.time() > deadline - 5:
                    break
                if new_adj[a][b] == -1:
                    for c3 in range(num_colors):
                        if check_rainbow(new_adj, a, b, c3):
                            new_adj[a][b] = c3
                            new_adj[b][a] = c3
                            new_neighbors[a].append(b)
                            new_neighbors[b].append(a)
                            new_edges.append([a, b, c3])
                            break
            
            if len(new_edges) > len(best_edges):
                best_edges = new_edges
                best_adj = new_adj
                best_neighbors = new_neighbors
            else:
                # Restore original
                best_adj[u][v] = c
                best_adj[v][u] = c
                best_neighbors[u].append(v)
                best_neighbors[v].append(u)
                break
        
        return best_edges

    best_edges = []
    best_count = 0
    
    # Strategy 1: Random greedy construction with different edge orderings
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        
        # Random edge ordering
        random_edge_order = all_edges[:]
        random.shuffle(random_edge_order)
        
        # Random color ordering
        def random_color_order():
            colors = list(range(num_colors))
            random.shuffle(colors)
            return colors
        
        edges, adj, neighbors = try_greedy_construction(random_edge_order, random_color_order)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
        
        # Try local improvement
        if time.time() < deadline - 30 and len(edges) >= best_count - 10:
            improved = try_remove_and_readd(adj, neighbors, edges)
            if len(improved) > best_count:
                best_count = len(improved)
                best_edges = improved[:]
    
    # Strategy 2: Structured construction - try to build a complete graph with specific color patterns
    # For a complete graph K_n, we need to color edges so that every 4-cycle is rainbow.
    # This is related to proper edge colorings and 1-factorizations.
    
    if time.time() < deadline - 20:
        # Try a structured approach: use a 1-factorization-like coloring
        # For n=20, we can try to use colors 0-19 in a structured way
        
        # One approach: for each color c, the edges of color c should form a matching
        # (no two edges of the same color share a vertex), because if they did,
        # we might create a non-rainbow 4-cycle.
        
        # Actually, for rainbow 4-cycles, we need that no 4-cycle has two edges of the same color.
        # This is weaker than requiring each color class to be a matching.
        
        # Let's try: assign colors based on vertex pairs using a formula
        # color(u,v) = (u + v) % 20 or similar
        
        for seed_offset in range(5):
            if time.time() > deadline - 10:
                break
            
            adj_matrix = [[-1]*n for _ in range(n)]
            neighbors = [[] for _ in range(n)]
            current_edges = []
            
            # Try different color assignment functions
            if seed_offset == 0:
                color_fn = lambda u, v: (u + v) % num_colors
            elif seed_offset == 1:
                color_fn = lambda u, v: (u * 3 + v * 7) % num_colors
            elif seed_offset == 2:
                color_fn = lambda u, v: (u * 5 + v * 11) % num_colors
            elif seed_offset == 3:
                color_fn = lambda u, v: (u * 7 + v * 13) % num_colors
            else:
                color_fn = lambda u, v: (u * 11 + v * 17) % num_colors
            
            for u, v in all_edges:
                c = color_fn(u, v)
                if check_rainbow(adj_matrix, u, v, c):
                    adj_matrix[u][v] = c
                    adj_matrix[v][u] = c
                    neighbors[u].append(v)
                    neighbors[v].append(u)
                    current_edges.append([u, v, c])
            
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = current_edges[:]
    
    return best_edges
