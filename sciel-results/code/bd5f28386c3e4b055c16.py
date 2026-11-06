import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2
DEADLINE = 85.0  # Safety margin before 90s kill

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def count_used(colors):
    return len(set(colors))

def is_valid_coloring(colors):
    """Check if no 4-cycle has four distinct edge colors."""
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Three 4-cycles on {a,b,c,d}:
                    # 1: a-b-c-d-a
                    c1 = (colors[edge_idx(a,b)], colors[edge_idx(b,c)],
                          colors[edge_idx(c,d)], colors[edge_idx(d,a)])
                    if len(set(c1)) == 4:
                        return False
                    # 2: a-b-d-c-a
                    c2 = (colors[edge_idx(a,b)], colors[edge_idx(b,d)],
                          colors[edge_idx(d,c)], colors[edge_idx(c,a)])
                    if len(set(c2)) == 4:
                        return False
                    # 3: a-c-b-d-a
                    c3 = (colors[edge_idx(a,c)], colors[edge_idx(c,b)],
                          colors[edge_idx(b,d)], colors[edge_idx(d,a)])
                    if len(set(c3)) == 4:
                        return False
    return True

def check_edge_recolor(colors, ei, ej, new_color):
    """Check if recoloring edge (ei,ej) to new_color maintains validity.
    Only need to check 4-cycles that include this edge."""
    ei, ej = min(ei, ej), max(ei, ej)
    # A 4-cycle containing edge (ei,ej) has form: ei - ej - x - y - ei
    # where x, y are distinct vertices not equal to ei or ej
    # The three cycles on {ei, ej, x, y} that use edge (ei,ej):
    # Actually, for any 4 vertices a<b<c<d, there are 3 cycles.
    # Edge (ei,ej) appears in some of these cycles.
    # Simpler: for each pair (x,y) from remaining vertices, check all 3 cycles
    # that include edge (ei,ej).
    
    other = [v for v in range(N) if v != ei and v != ej]
    for xi in range(len(other)):
        for xj in range(xi + 1, len(other)):
            x = other[xi]
            y = other[xj]
            # 4 vertices: ei, ej, x, y
            # Sort them
            verts = sorted([ei, ej, x, y])
            a, b, c, d = verts
            # Check all 3 cycles, but only those that include edge (ei,ej)
            # Cycle 1: a-b-c-d-a
            cyc1_edges = [(a,b), (b,c), (c,d), (d,a)]
            if (ei,ej) in cyc1_edges or (ej,ei) in [(u,v) for u,v in cyc1_edges]:
                cols = [colors[edge_idx(u,v)] if (u,v) != (ei,ej) and (u,v) != (ej,ei) else new_color
                        for u,v in cyc1_edges]
                if len(set(cols)) == 4:
                    return False
            # Cycle 2: a-b-d-c-a
            cyc2_edges = [(a,b), (b,d), (d,c), (c,a)]
            if (ei,ej) in cyc2_edges or (ej,ei) in [(u,v) for u,v in cyc2_edges]:
                cols = [colors[edge_idx(u,v)] if (u,v) != (ei,ej) and (u,v) != (ej,ei) else new_color
                        for u,v in cyc2_edges]
                if len(set(cols)) == 4:
                    return False
            # Cycle 3: a-c-b-d-a
            cyc3_edges = [(a,c), (c,b), (b,d), (d,a)]
            if (ei,ej) in cyc3_edges or (ej,ei) in [(u,v) for u,v in cyc3_edges]:
                cols = [colors[edge_idx(u,v)] if (u,v) != (ei,ej) and (u,v) != (ej,ei) else new_color
                        for u,v in cyc3_edges]
                if len(set(cols)) == 4:
                    return False
    return True

def construct():
    start_time = time.time()
    
    # Start with min(i,j) coloring - uses 15 colors (0-14)
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            colors[edge_idx(i, j)] = i
    
    used_colors = set(colors)
    current_count = count_used(colors)
    
    # Precompute for faster checking: for each edge, list of (x,y) pairs to check
    # Actually, let's just do greedy recoloring
    
    # Strategy: try to recolor edges to use new colors
    # Process edges in an order that's likely to allow more recolorings
    # Edges incident to vertex 15 first, then 14, etc.
    
    # Create list of edges to try, sorted by "difficulty"
    # Edges with higher min(i,j) are "higher" and may be easier to recolor
    edges_to_try = []
    for i in range(N):
        for j in range(i + 1, N):
            edges_to_try.append((i, j))
    
    # Sort: try edges incident to higher-index vertices first
    # Actually, let's try a specific order: edges (i, 15), (i, 14), etc.
    ordered_edges = []
    for high in range(N - 1, -1, -1):
        for low in range(high - 1, -1, -1):
            ordered_edges.append((low, high))
    
    next_new_color = max(used_colors) + 1 if used_colors else 0
    max_color = 119
    
    # Greedy: for each edge in order, try to assign a new color
    for ei, ej in ordered_edges:
        if time.time() > DEADLINE - 5:
            break
        if next_new_color > max_color:
            break
        
        idx = edge_idx(ei, ej)
        old_color = colors[idx]
        
        # Try new color
        if next_new_color != old_color:
            if check_edge_recolor(colors, ei, ej, next_new_color):
                colors[idx] = next_new_color
                used_colors.add(next_new_color)
                next_new_color += 1
    
    # Second pass: try to recolor more edges with remaining new colors
    # Also try recoloring edges that currently have old colors to new colors
    for ei, ej in ordered_edges:
        if time.time() > DEADLINE - 5:
            break
        if next_new_color > max_color:
            break
        
        idx = edge_idx(ei, ej)
        old_color = colors[idx]
        
        if next_new_color != old_color:
            if check_edge_recolor(colors, ei, ej, next_new_color):
                colors[idx] = next_new_color
                used_colors.add(next_new_color)
                next_new_color += 1
    
    # Third pass: try all edges with all remaining colors
    for ei, ej in ordered_edges:
        if time.time() > DEADLINE - 5:
            break
        
        idx = edge_idx(ei, ej)
        old_color = colors[idx]
        
        for c in range(next_new_color, max_color + 1):
            if c == old_color:
                continue
            if check_edge_recolor(colors, ei, ej, c):
                colors[idx] = c
                used_colors.add(c)
                # Don't increment next_new_color since we're using arbitrary colors
                break
    
    # Final: try to add any remaining colors on any edge
    remaining_colors = set(range(0, max_color + 1)) - used_colors
    for c in sorted(remaining_colors):
        if time.time() > DEADLINE - 3:
            break
        
        found = False
        for ei in range(N):
            if found:
                break
            for ej in range(ei + 1, N):
                if time.time() > DEADLINE - 3:
                    break
                idx = edge_idx(ei, ej)
                if colors[idx] == c:
                    continue
                if check_edge_recolor(colors, ei, ej, c):
                    colors[idx] = c
                    used_colors.add(c)
                    found = True
                    break
    
    # Verify the final coloring
    if not is_valid_coloring(colors):
        # Fallback to min(i,j) coloring
        for i in range(N):
            for j in range(i + 1, N):
                colors[edge_idx(i, j)] = i
    
    return colors
