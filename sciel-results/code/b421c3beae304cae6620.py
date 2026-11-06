import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n = 20
    num_colors = 20
    
    # We'll build a proper edge-coloring where each color is a matching.
    # Constraint: for any two colors (a,b), the subgraph of edges colored a or b must be C4-free.
    # 
    # Approach: Represent each color class as a set of edges (a matching).
    # color_edges[c] = set of (u,v) edges with color c
    # 
    # To check if adding edge (u,v) with color c is valid:
    #   - (u,v) must not conflict with existing edges of color c (proper coloring)
    #   - For every other color d, the 2-color subgraph {c, d} must remain C4-free
    #
    # A C4 in colors {c,d}: vertices a,b,c2,d2 where edges (a,b) and (c2,d2) have color c,
    # and (b,c2) and (d2,a) have color d. 
    # Equivalently: in the 2-color subgraph, we look for 4-cycles.
    #
    # Efficient check: When adding (u,v) with color c, for each color d != c:
    #   The new edge (u,v) in color c could form a C4 in {c,d} if there exist 
    #   vertices x,y such that:
    #     - (x,u) has color d, (v,y) has color d, (x,y) has color c
    #     - or (x,v) has color d, (u,y) has color d, (x,y) has color c
    #   Wait, let me think more carefully.
    #
    # A C4 in the 2-color subgraph {c,d} using the new edge (u,v) of color c:
    #   The C4 has 4 vertices. One edge is (u,v) with color c.
    #   The opposite edge must also have color c (since in a C4 with 2 colors, 
    #   opposite edges have the same color). So we need another edge (x,y) with color c
    #   such that the other two edges (u,x) and (v,y) [or (u,y) and (v,x)] have color d.
    #
    # So: for each existing edge (x,y) with color c (x,y != u,v), check if:
    #   - (u,x) and (v,y) both have color d, OR
    #   - (u,y) and (v,x) both have color d
    #   for some color d.
    #
    # This is O(E_c * 2) per color d, and we have 19 other colors.
    # Total: O(E_c * 19 * 2) per edge addition.
    
    def try_build(seed):
        random.seed(seed)
        
        # color_edges[c] = list of (u,v) tuples
        color_edges = [[] for _ in range(num_colors)]
        # edge_color[(u,v)] = c for quick lookup
        edge_color = {}
        # adj_color[u][v] = c if edge exists, else -1
        adj_color = [[-1] * n for _ in range(n)]
        
        # For each color c, color_adj[c][u] = set of neighbors via color c
        color_adj = [set() for _ in range(num_colors)]
        # Also track: for each color c, which vertices are matched
        color_matched = [set() for _ in range(num_colors)]
        
        def can_add_edge(u, v, c):
            # Check proper coloring: u and v not already matched in color c
            if u in color_matched[c] or v in color_matched[c]:
                return False
            
            # Check no C4 in any 2-color subgraph {c, d}
            # For each existing edge (x,y) in color c, check if adding (u,v) creates a C4
            for (x, y) in color_edges[c]:
                # Case 1: (u,x) and (v,y) both in color d for some d
                # Case 2: (u,y) and (v,x) both in color d for some d
                # We need to check if there exists a color d such that both edges have color d
                
                # Get colors of (u,x), (v,y)
                c_ux = adj_color[u][x] if u != x else -1
                c_vy = adj_color[v][y] if v != y else -1
                if c_ux != -1 and c_ux == c_vy and c_ux != c:
                    return False  # C4 with colors c and c_ux
                
                c_uy = adj_color[u][y] if u != y else -1
                c_vx = adj_color[v][x] if v != x else -1
                if c_uy != -1 and c_uy == c_vx and c_uy != c:
                    return False  # C4 with colors c and c_uy
            
            return True
        
        def add_edge(u, v, c):
            color_edges[c].append((u, v))
            edge_color[(u, v)] = c
            edge_color[(v, u)] = c
            adj_color[u][v] = c
            adj_color[v][u] = c
            color_adj[c].add((u, v))
            color_adj[c].add((v, u))
            color_matched[c].add(u)
            color_matched[c].add(v)
        
        def get_edge_count():
            return sum(len(e) for e in color_edges)
        
        # Greedy: try to add edges
        # Order: try all (u,v,c) combinations in random order
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        # Multiple passes to try different color assignments
        for pass_num in range(10):
            if time.time() > deadline - 5:
                break
            
            random.shuffle(all_pairs)
            colors_order = list(range(num_colors))
            random.shuffle(colors_order)
            
            for u, v in all_pairs:
                if time.time() > deadline - 5:
                    break
                if adj_color[u][v] != -1:
                    continue
                # Try each available color
                for c in colors_order:
                    if u in color_matched[c] or v in color_matched[c]:
                        continue
                    if can_add_edge(u, v, c):
                        add_edge(u, v, c)
                        break
        
        return get_edge_count()
    
    # Try multiple seeds
    best_count = 0
    best_seed = 0
    
    for seed in range(50000):
        if time.time() > deadline - 10:
            break
        count = try_build(seed)
        if count > best_count:
            best_count = count
            best_seed = seed
    
    # Rebuild with best seed
    random.seed(best_seed)
    color_edges = [[] for _ in range(num_colors)]
    edge_color = {}
    adj_color = [[-1] * n for _ in range(n)]
    color_adj = [set() for _ in range(num_colors)]
    color_matched = [set() for _ in range(num_colors)]
    
    def can_add_edge(u, v, c):
        if u in color_matched[c] or v in color_matched[c]:
            return False
        for (x, y) in color_edges[c]:
            c_ux = adj_color[u][x] if u != x else -1
            c_vy = adj_color[v][y] if v != y else -1
            if c_ux != -1 and c_ux == c_vy and c_ux != c:
                return False
            c_uy = adj_color[u][y] if u != y else -1
            c_vx = adj_color[v][x] if v != x else -1
            if c_uy != -1 and c_uy == c_vx and c_uy != c:
                return False
        return True
    
    def add_edge(u, v, c):
        color_edges[c].append((u, v))
        edge_color[(u, v)] = c
        edge_color[(v, u)] = c
        adj_color[u][v] = c
        adj_color[v][u] = c
        color_adj[c].add((u, v))
        color_adj[c].add((v, u))
        color_matched[c].add(u)
        color_matched[c].add(v)
    
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    for pass_num in range(10):
        if time.time() > deadline - 2:
            break
        random.shuffle(all_pairs)
        colors_order = list(range(num_colors))
        random.shuffle(colors_order)
        for u, v in all_pairs:
            if time.time() > deadline - 2:
                break
            if adj_color[u][v] != -1:
                continue
            for c in colors_order:
                if u in color_matched[c] or v in color_matched[c]:
                    continue
                if can_add_edge(u, v, c):
                    add_edge(u, v, c)
                    break
    
    # Build result
    edges = []
    for c in range(num_colors):
        for (u, v) in color_edges[c]:
            edges.append([u, v, c])
    
    return edges

