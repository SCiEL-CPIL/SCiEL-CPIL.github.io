"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 80  # safety margin
    N = 16
    n_edges = N * (N - 1) // 2
    
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * N - i * (i + 1) // 2 + (j - i - 1)
    
    # Precompute all C4s: for each 4-vertex subset, the 3 cycles
    # A C4 on {a,b,c,d} (sorted) has cycles:
    # (a,b,c,d): edges (a,b),(b,c),(c,d),(d,a)
    # (a,b,d,c): edges (a,b),(b,d),(d,c),(c,a)
    # (a,c,b,d): edges (a,c),(c,b),(b,d),(d,a)
    import itertools
    c4_list = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        c4_list.append((edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)))
        # Cycle 2: a-b-d-c-a
        c4_list.append((edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)))
        # Cycle 3: a-c-b-d-a
        c4_list.append((edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)))
    
    num_c4 = len(c4_list)
    
    # For each edge, precompute which C4s it belongs to
    edge_to_c4s = [[] for _ in range(n_edges)]
    for idx, c4 in enumerate(c4_list):
        for e in c4:
            edge_to_c4s[e].append(idx)
    
    # Initial coloring: use a block structure
    # Group vertices into 4 groups of 4
    # Color based on group pair
    def initial_coloring():
        colors = [0] * n_edges
        for i in range(N):
            for j in range(i+1, N):
                gi = i // 4
                gj = j // 4
                if gi == gj:
                    colors[edge_idx(i,j)] = 0
                else:
                    mg = min(gi, gj)
                    Mg = max(gi, gj)
                    colors[edge_idx(i,j)] = 4 * mg + Mg
        return colors
    
    best_colors = initial_coloring()
    best_count = len(set(best_colors))
    
    # Try to improve by changing edge colors to new colors
    def is_valid(colors):
        for c4 in c4_list:
            c0, c1, c2, c3 = colors[c4[0]], colors[c4[1]], colors[c4[2]], colors[c4[3]]
            if len({c0, c1, c2, c3}) == 4:
                return False
        return True
    
    # Verify initial
    if not is_valid(best_colors):
        # Fallback to all zeros
        return [0] * n_edges
    
    # Local search: try to recolor edges to unused colors
    used_colors = set(best_colors)
    
    # Precompute: for each C4, which edges are in it (for fast constraint checking)
    # When we change edge e to color c, we need to check all C4s containing e
    # For each such C4, the other 3 edges must not all have distinct colors from c
    
    for iteration in range(200000):
        if time.time() > deadline - 5:
            break
        
        # Pick a random edge
        e = random.randint(0, n_edges - 1)
        old_color = best_colors[e]
        
        # Try to assign a new color not in used_colors
        # Or try to free up a color
        new_color = None
        # Try unused colors first
        for c in range(120):
            if c not in used_colors:
                new_color = c
                break
        
        if new_color is None:
            # All colors used, try to recolor to a color that's used elsewhere
            # to potentially free up another color later
            # For now, skip
            continue
        
        # Check if changing e to new_color violates any C4
        valid = True
        for c4_idx in edge_to_c4s[e]:
            c4 = c4_list[c4_idx]
            # Get the other 3 edges
            other_edges = [c4[k] for k in range(4) if c4[k] != e]
            other_colors = {best_colors[oe] for oe in other_edges}
            # After change, colors in this C4: new_color + other_colors
            if len(other_colors | {new_color}) == 4:
                valid = False
                break
        
        if valid:
            best_colors[e] = new_color
            used_colors.add(new_color)
            best_count = len(used_colors)
        else:
            # Try to find a color that works (might be an already-used color)
            # This doesn't increase count but might enable future improvements
            # Skip for now to keep it simple
            pass
    
    # Final verification
    if not is_valid(best_colors):
        return [0] * n_edges
    
    return best_colors
# EVOLVE-BLOCK-END
