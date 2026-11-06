import numpy as np
import random

def construct():
    n = 20
    all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    best_edges = 0
    best_graph = []
    
    # Number of trials
    trials = 100
    
    for _ in range(trials):
        random.shuffle(all_pairs)
        
        # adj[u][v] stores the color of edge (u,v) or -1 if no edge
        adj = np.full((n, n), -1, dtype=np.int8)
        edges = []
        
        for u, v in all_pairs:
            # Find all 4-cycles that would be completed by adding (u, v)
            # Cycle: u-v-x-y-u
            # Requires edges (v,x), (x,y), (y,u)
            
            # Get neighbors of u and v
            nu = np.where(adj[u] != -1)[0]
            nv = np.where(adj[v] != -1)[0]
            
            forbidden_colors = set()
            
            # Check all potential x in N(v) and y in N(u)
            # Note: x and y must be distinct from u and v
            # adj[u, x] != -1 implies x != u (since adj[u,u] is -1)
            # adj[v, y] != -1 implies y != v
            
            for x in nv:
                # x is a neighbor of v
                # Edge (v, x) exists with color adj[v, x]
                c_vx = adj[v, x]
                
                for y in nu:
                    # y is a neighbor of u
                    # Edge (u, y) exists with color adj[u, y]
                    c_uy = adj[u, y]
                    
                    # Check if (x, y) is an edge
                    if x == y:
                        continue # Not a 4-cycle, it's a triangle or path
                    if adj[x, y] != -1:
                        c_xy = adj[x, y]
                        
                        # The cycle is u-v-x-y-u
                        # Colors: c_uv (new), c_vx, c_xy, c_uy
                        # We need {c_uv, c_vx, c_xy, c_uy} to be distinct.
                        # So c_uv cannot be c_vx, c_xy, or c_uy.
                        forbidden_colors.add(c_vx)
                        forbidden_colors.add(c_xy)
                        forbidden_colors.add(c_uy)
            
            # Find a valid color
            valid_color = -1
            for c in range(20):
                if c not in forbidden_colors:
                    valid_color = c
                    break
            
            if valid_color != -1:
                adj[u, v] = valid_color
                adj[v, u] = valid_color
                edges.append([u, v, valid_color])
        
        if len(edges) > best_edges:
            best_edges = len(edges)
            best_graph = edges
            
    return best_graph
