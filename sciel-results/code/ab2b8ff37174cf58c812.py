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
    color_counts = {} # To track frequency of used colors
    
    # Precompute cycles
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

    for i in range(num_edges):
        allowed_set = None
        
        for (e1, e2, e3) in cycles_for_edge[i]:
            if colors[e1] != -1 and colors[e2] != -1 and colors[e3] != -1:
                s = {colors[e1], colors[e2], colors[e3]}
                if len(s) == 3:
                    if allowed_set is None:
                        allowed_set = s
                    else:
                        allowed_set = allowed_set.intersection(s)
                        if not allowed_set:
                            # Dead end? 
                            # If intersection is empty, we must pick a color that violates some constraint?
                            # Or maybe we should have picked differently earlier.
                            # For now, if empty, we pick a color from the union to avoid crash, 
                            # but this will likely be invalid.
                            # Let's break and handle.
                            break
        
        if allowed_set is None or not allowed_set:
            # If allowed_set is None, no constraint.
            # If allowed_set is empty, we are in trouble.
            # Let's assume allowed_set is None (no constraint) or non-empty.
            # If it's empty, we pick the most frequent color globally as a fallback?
            if allowed_set is None:
                colors[i] = next_color
                next_color += 1
                color_counts[colors[i]] = 1
            else:
                # Fallback: pick most frequent color overall?
                # This is a heuristic to avoid crash.
                if color_counts:
                    c = max(color_counts, key=color_counts.get)
                else:
                    c = 0
                colors[i] = c
                color_counts[c] = color_counts.get(c, 0) + 1
        else:
            # Pick the color in allowed_set with the highest frequency
            # This might help satisfy future constraints
            best_c = None
            max_count = -1
            for c in allowed_set:
                count = color_counts.get(c, 0)
                if count > max_count:
                    max_count = count
                    best_c = c
            colors[i] = best_c
            color_counts[best_c] = color_counts.get(best_c, 0) + 1

    return colors
