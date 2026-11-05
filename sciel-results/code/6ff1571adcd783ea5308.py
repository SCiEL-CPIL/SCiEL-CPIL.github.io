def dpll(colors, edge_counts):
    # ... propagation ...
    for v in range(n):
        if colors[v] == -1:
            # Try 0
            new_colors = colors[:]
            new_colors[v] = 0
            new_counts = [list(c) for c in edge_counts]
            for ei in vertex_edges[v]:
                new_counts[ei][0] += 1
            if not dpll(new_colors, new_counts):
                return False
            
            # Try 1
            new_colors = colors[:]
            new_colors[v] = 1
            new_counts = [list(c) for c in edge_counts]
            for ei in vertex_edges[v]:
                new_counts[ei][1] += 1
            if not dpll(new_colors, new_counts):
                return False
            
            return True
    return False
