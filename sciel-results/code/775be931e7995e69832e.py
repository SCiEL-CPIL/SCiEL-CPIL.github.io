"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a list of 120 integer colors for the edges of K_16.
    The edges are ordered lexicographically by (i, j) with 0 <= i < j < 16.
    
    Uses a local search approach:
    1. Start with base coloring c(i,j) = i (15 colors, known valid).
    2. Try to recolor edges to unused colors to increase the count.
    3. Validate that no 4-cycle has 4 distinct edge colors.
    """
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    # Build edge list and index mapping
    edges = []
    edge_index = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_index[(i, j)] = idx
            edges.append((i, j))
            idx += 1
    
    num_edges = len(edges)  # 120
    
    # Precompute all 4-cycles (all three cycle structures on each 4-vertex subset)
    # A 4-cycle on vertices {a,b,c,d} can be:
    #   a-b-c-d-a, a-b-d-c-a, a-c-b-d-a
    # We need to check all C4s in K_16.
    # For efficiency, precompute the edge indices for each C4.
    
    all_c4s = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Three 4-cycles on these 4 vertices:
        # Cycle 1: a-b-c-d-a
        c4_1 = [(a,b), (b,c), (c,d), (d,a)]
        # Cycle 2: a-b-d-c-a
        c4_2 = [(a,b), (b,d), (d,c), (c,a)]
        # Cycle 3: a-c-b-d-a
        c4_3 = [(a,c), (c,b), (b,d), (d,a)]
        
        for c4 in [c4_1, c4_2, c4_3]:
            edge_idxs = []
            valid = True
            for (u, v) in c4:
                if u < v:
                    edge_idxs.append(edge_index[(u, v)])
                else:
                    edge_idxs.append(edge_index[(v, u)])
            all_c4s.append(edge_idxs)
    
    # Start with base coloring: c(i,j) = i
    colors = [0] * num_edges
    for idx, (i, j) in enumerate(edges):
        colors[idx] = i
    
    # Track which colors are in use
    colors_in_use = set(colors)
    next_color = max(colors_in_use) + 1 if colors_in_use else 0
    
    # Local search: try to recolor edges to new colors
    # For each edge, try assigning a new color and check if any C4 becomes rainbow
    # To be efficient, for each edge, precompute which C4s contain it
    
    # Build edge -> list of C4 indices containing it
    edge_to_c4s = [[] for _ in range(num_edges)]
    for c4_idx, edge_idxs in enumerate(all_c4s):
        for eidx in edge_idxs:
            edge_to_c4s[eidx].append(c4_idx)
    
    # Function to check if a C4 is rainbow
    def is_rainbow_c4(c4_edge_idxs, col):
        c = set(col[e] for e in c4_edge_idxs)
        return len(c) == 4
    
    # Local search iterations
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for eidx in range(num_edges):
            if (time.time() - start_time) > deadline:
                break
            # Try to recolor this edge to a new color
            new_color = next_color
            old_color = colors[eidx]
            
            # Check all C4s containing this edge
            valid = True
            for c4_idx in edge_to_c4s[eidx]:
                # Temporarily set new color
                colors[eidx] = new_color
                if is_rainbow_c4(all_c4s[c4_idx], colors):
                    valid = False
                    break
                colors[eidx] = old_color
            
            if valid:
                colors[eidx] = new_color
                colors_in_use.add(new_color)
                next_color += 1
                improved = True
    
    # Also try: can we add more colors by recoloring edges that currently share
    # a color with many others? Try to split colors.
    # For each color group, try to move one edge to a new color.
    if (time.time() - start_time) < deadline:
        # Group edges by color
        color_groups = {}
        for eidx in range(num_edges):
            c = colors[eidx]
            if c not in color_groups:
                color_groups[c] = []
            color_groups[c].append(eidx)
        
        # Try to recolor one edge from each group to a new color
        for c, group in sorted(color_groups.items(), key=lambda x: -len(x[1])):
            if (time.time() - start_time) > deadline:
                break
            for eidx in group:
                new_color = next_color
                old_color = colors[eidx]
                
                valid = True
                for c4_idx in edge_to_c4s[eidx]:
                    colors[eidx] = new_color
                    if is_rainbow_c4(all_c4s[c4_idx], colors):
                        valid = False
                        break
                    colors[eidx] = old_color
                
                if valid:
                    colors[eidx] = new_color
                    colors_in_use.add(new_color)
                    next_color += 1
                    # Remove from old group if empty
                    group.remove(eidx)
                    break
    
    return colors
# EVOLVE-BLOCK-END
