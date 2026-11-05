"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    n = 20
    best_edges = []
    best_count = 0
    
    # Greedy approach: build graph edge by edge, checking validity
    # For each potential edge, try adding it with some color and verify all 4-cycles remain rainbow
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # We'll use a greedy strategy with random restarts
    # For each restart, shuffle edge order and greedily add edges
    
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(restart * 12345 + 7)
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        # Track current graph
        adj = {i: set() for i in range(n)}  # adjacency sets
        edge_colors = {}  # (u,v) -> color where u < v
        current_edges = []
        
        for u, v in shuffled_edges:
            if time.time() > deadline:
                break
            
            # Try to find a valid color for this edge
            valid_color = None
            for c in range(n):
                # Tentatively add edge
                adj[u].add(v)
                adj[v].add(u)
                edge_colors[(u, v)] = c
                
                # Check all 4-cycles involving this edge
                # A 4-cycle containing edge (u,v): u-w-x-v-u or u-v-x-w-u
                # Actually, 4-cycles containing edge (u,v) are: u-v-a-b-u where a,b are common neighbors
                # Wait, that's not right. Let me think again.
                # A 4-cycle is u1-u2-u3-u4-u1. If it contains edge (u,v), then either:
                # - u and v are adjacent in the cycle: u-v-x-y-u or v-u-x-y-v
                # So we need to check all pairs (x,y) where x is neighbor of v, y is neighbor of u,
                # and x != y, and x is connected to y, and the 4 edges have distinct colors.
                
                valid = True
                # For each common "path" of length 2 from u to v through other vertices
                # Actually, simpler: iterate over all pairs of vertices (x,y) where
                # x is adjacent to v and y is adjacent to u, x != y, x != u, y != v,
                # and check if u-x-y-v-u forms a 4-cycle (i.e., x-y is an edge)
                
                for x in adj[v]:
                    if x == u:
                        continue
                    for y in adj[u]:
                        if y == v or y == x:
                            continue
                        if x in adj[y]:
                            # Check 4-cycle: u-v-x-y-u
                            # Edges: (u,v), (v,x), (x,y), (y,u)
                            c_uv = edge_colors.get((min(u,v), max(u,v)))
                            c_vx = edge_colors.get((min(v,x), max(v,x)))
                            c_xy = edge_colors.get((min(x,y), max(x,y)))
                            c_yu = edge_colors.get((min(y,u), max(y,u)))
                            
                            if c_uv is None or c_vx is None or c_xy is None or c_yu is None:
                                continue
                            
                            colors = [c_uv, c_vx, c_xy, c_yu]
                            if len(set(colors)) != 4:
                                valid = False
                                break
                    if not valid:
                        break
                    if time.time() > deadline:
                        break
                
                if valid:
                    valid_color = c
                    break
                else:
                    # Remove tentative edge
                    adj[u].remove(v)
                    adj[v].remove(u)
                    del edge_colors[(u, v)]
            
            if valid_color is not None:
                current_edges.append([u, v, valid_color])
            
            if len(current_edges) > best_count:
                best_edges = current_edges[:]
                best_count = len(current_edges)
        
        if len(current_edges) > best_count:
            best_edges = current_edges[:]
            best_count = len(current_edges)
    
    return best_edges

# EVOLVE-BLOCK-END
