import numpy as np

N = 16

def construct():
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            edges.append((u, v))
            
    num_edges = len(edges)
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i
        
    colors = [-1] * num_edges
    next_color = 0
    
    # Precompute cycles for each edge
    # For an edge (u,v), the 4-cycles containing it are formed by pairs (x,y) of other vertices.
    # The two cycles are:
    # 1. u-v-x-y-u  (edges: uv, vx, xy, yu)
    # 2. u-v-y-x-u  (edges: uv, vy, yx, xu)
    
    cycles_for_edge = [[] for _ in range(num_edges)]
    for i, (u, v) in enumerate(edges):
        others = [k for k in range(N) if k != u and k != v]
        for idx_x in range(len(others)):
            x = others[idx_x]
            for idx_y in range(idx_x + 1, len(others)):
                y = others[idx_y]
                # Cycle 1: u-v-x-y-u
                e1 = edge_to_idx[(v, x)]
                e2 = edge_to_idx[(x, y)]
                e3 = edge_to_idx[(y, u)]
                cycles_for_edge[i].append((e1, e2, e3))
                
                # Cycle 2: u-v-y-x-u
                e1 = edge_to_idx[(v, y)]
                e2 = edge_to_idx[(y, x)]
                e3 = edge_to_idx[(x, u)]
                cycles_for_edge[i].append((e1, e2, e3))
                
    # Greedy Coloring
    for i in range(num_edges):
        # Identify constraints from cycles where the other 3 edges are already colored
        # If a cycle has 3 distinct colors among the other 3 edges, the current edge MUST take one of those colors.
        # The valid colors are the intersection of all such constraint sets.
        
        allowed_set = None
        
        for (e1, e2, e3) in cycles_for_edge[i]:
            # Only consider cycles where all other 3 edges are colored
            if colors[e1] != -1 and colors[e2] != -1 and colors[e3] != -1:
                s = {colors[e1], colors[e2], colors[e3]}
                if len(s) == 3:
                    if allowed_set is None:
                        allowed_set = s
                    else:
                        allowed_set = allowed_set.intersection(s)
                        if not allowed_set:
                            break
        
        if allowed_set is not None:
            # Must pick a color from allowed_set
            # To maximize the number of distinct colors used, we should prefer a color that is already in use
            # over a new one? No, we are forced to pick from allowed_set.
            # However, if we pick a color, it might affect future constraints.
            # A simple heuristic is to pick the smallest color to keep the palette compact,
            # but we want to MAXIMIZE the number of colors.
            # Actually, if we are forced, we can't add a new color.
            # So we just pick one. Which one?
            # Picking any is valid. Let's pick the smallest for determinism.
            colors[i] = min(allowed_set)
        else:
            # No constraint forces us to reuse a color.
            # We can use a new color to maximize the count.
            colors[i] = next_color
            next_color += 1
            
    return colors
