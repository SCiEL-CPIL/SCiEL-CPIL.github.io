"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import itertools
import numpy as np

N = 16
E = N*(N-1)//2  # 120 edges

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i*(2*N-1-i)//2 + j - i - 1

def precompute_cycles():
    """
    Returns:
    - cycles: list of tuples of 4 edge indices forming a 4-cycle
    - edge_to_cycles: for each edge, list of cycle indices it belongs to
    """
    cycles = []
    edge_to_cycles = [[] for _ in range(E)]
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b, b-c, c-d, d-a
        e1 = (edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a))
        # Cycle 2: a-b, b-d, d-c, c-a
        e2 = (edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a))
        # Cycle 3: a-c, c-b, b-d, d-a
        e3 = (edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a))
        
        for cyc in (e1, e2, e3):
            ci = len(cycles)
            cycles.append(cyc)
            for e in cyc:
                edge_to_cycles[e].append(ci)
    return cycles, edge_to_cycles

def count_colors(colors):
    return len(set(colors))

def check_valid(colors, cycles):
    """Check if no 4-cycle has 4 distinct colors."""
    for cyc in cycles:
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def construct():
    # Safety margin: 60s limit, kill at 90s. Use 50s for computation.
    deadline = time.time() + 50.0
    
    cycles, edge_to_cycles = precompute_cycles()
    
    # Start with a valid construction: color(i,j) = i (min vertex)
    # This uses 15 colors (0..14).
    colors = np.zeros(E, dtype=int)
    for i in range(N):
        for j in range(i+1, N):
            e = edge_idx(i, j)
            colors[e] = i
    
    best = colors.copy()
    best_count = count_colors(best)
    
    random.seed(42)
    max_color = 119
    
    # Precompute: for each edge, which other edges are in cycles with it
    # and the positions they occupy in those cycles
    # edge_conflict[e] = list of (other_edge, cycle_idx, position_in_cycle)
    edge_conflict = [[] for _ in range(E)]
    for ci, cyc in enumerate(cycles):
        for pos in range(4):
            e = cyc[pos]
            for other_pos in range(4):
                if other_pos != pos:
                    other_e = cyc[other_pos]
                    edge_conflict[e].append((other_e, ci, other_pos))
    
    # Greedy assignment with constraint awareness
    # Process edges in order of decreasing number of cycle partners
    edge_order = sorted(range(E), key=lambda e: -len(edge_to_cycles[e]))
    
    # For each edge, track which colors are "forbidden" because they would
    # create a rainbow cycle given current assignments of other edges in those cycles
    # We'll use a different strategy: try to assign new colors while checking constraints
    
    # Maintain a set of used colors for quick lookup
    used_colors = set(colors.tolist())
    
    # Local search with smarter move selection
    # Key insight: to add a new color, we need an edge where:
    # 1. The new color doesn't appear in any cycle with 3 other distinct colors
    # 2. Prefer edges with fewer constraints (fewer cycles)
    
    # Sort edges by number of cycles (fewer first - easier to reassign)
    edge_order_easy = sorted(range(E), key=lambda e: len(edge_to_cycles[e]))
    
    while time.time() < deadline:
        # Pick an edge from the easy-to-reassign set
        e = random.choice(edge_order_easy)
        old_color = colors[e]
        
        # Find valid new colors for this edge
        # A color c is valid if for every cycle containing e,
        # the other 3 edges don't have 3 distinct colors all different from c
        
        valid_colors = []
        for c in range(max_color + 1):
            if c == old_color:
                continue
            valid = True
            for ci in edge_to_cycles[e]:
                cyc = cycles[ci]
                # Get the other 3 edges in this cycle
                other_edges = [cyc[i] for i in range(4) if cyc[i] != e]
                other_colors = [colors[oe] for oe in other_edges]
                # If the 3 other colors are all distinct and none equals c, then c would create a rainbow
                if len(set(other_colors)) == 3 and c not in other_colors:
                    valid = False
                    break
            if valid:
                valid_colors.append(c)
        
        if not valid_colors:
            continue
        
        # Prefer colors not in used_colors (to maximize diversity)
        new_used = [c for c in valid_colors if c not in used_colors]
        if new_used:
            # Pick a random new color
            new_color = random.choice(new_used)
        else:
            # Pick a random valid color
            new_color = random.choice(valid_colors)
        
        # Apply the move
        colors[e] = new_color
        used_colors.add(new_color)
        if old_color not in colors:
            used_colors.discard(old_color)
        
        nc = count_colors(colors)
        if nc > best_count:
            best = colors.copy()
            best_count = nc
    
    return best.tolist()
# EVOLVE-BLOCK-END
