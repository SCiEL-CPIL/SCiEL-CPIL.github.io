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
    - cycle_other_edges: for each cycle and position, the other 3 edge indices
    """
    cycles = []
    edge_to_cycles = [[] for _ in range(E)]
    cycle_other_edges = []  # cycle_other_edges[ci][pos] = tuple of 3 other edge indices
    
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
            other_edges_for_cycle = []
            for pos in range(4):
                others = tuple(cyc[p] for p in range(4) if p != pos)
                other_edges_for_cycle.append(others)
            cycle_other_edges.append(tuple(other_edges_for_cycle))
            for e in cyc:
                edge_to_cycles[e].append(ci)
    return cycles, edge_to_cycles, cycle_other_edges

def count_colors(colors):
    return len(set(colors))

def construct():
    # Safety margin: 60s limit, kill at 90s. Use 55s for computation.
    deadline = time.time() + 55.0
    
    cycles, edge_to_cycles, cycle_other_edges = precompute_cycles()
    
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
    
    # Precompute for each edge: list of (cycle_idx, position_in_cycle)
    edge_cycle_pos = [[] for _ in range(E)]
    for ci, cyc in enumerate(cycles):
        for pos in range(4):
            e = cyc[pos]
            edge_cycle_pos[e].append((ci, pos))
    
    # Sort edges by number of cycles (fewer first - easier to reassign)
    edge_order_easy = sorted(range(E), key=lambda e: len(edge_to_cycles[e]))
    
    # Track color counts for quick "is this color still used" check
    color_counts = np.zeros(max_color + 1, dtype=int)
    for c in colors:
        color_counts[c] += 1
    
    # For each edge, maintain a set of colors that are "forbidden"
    # A color c is forbidden for edge e if there exists a cycle containing e
    # where the other 3 edges have 3 distinct colors, none equal to c
    # We'll compute this on the fly but cache results
    
    # Precompute for each edge the list of (other_edges_tuple) for each cycle it's in
    # edge_constraint_data[e] = list of tuples of 3 edge indices (the other edges in each cycle)
    edge_constraint_data = []
    for e in range(E):
        constraints = []
        for ci, pos in edge_cycle_pos[e]:
            others = cycle_other_edges[ci][pos]
            constraints.append(others)
        edge_constraint_data.append(tuple(constraints))
    
    # Main search loop
    while time.time() < deadline:
        # Pick an edge from the easy-to-reassign set
        e = random.choice(edge_order_easy)
        old_color = colors[e]
        
        # Find valid new colors for this edge
        # A color c is valid if for every cycle containing e,
        # the other 3 edges don't have 3 distinct colors all different from c
        
        # Get the constraints for this edge
        constraints = edge_constraint_data[e]
        
        # Build set of colors that would be invalid
        invalid_colors = set()
        for others in constraints:
            c0 = colors[others[0]]
            c1 = colors[others[1]]
            c2 = colors[others[2]]
            # If all three are distinct, then any color not in {c0,c1,c2} would create a rainbow
            if c0 != c1 and c1 != c2 and c0 != c2:
                # Colors c0, c1, c2 are safe (they match one of the existing colors)
                # All other colors would create a rainbow
                # Instead of adding all invalid colors, note that only c0,c1,c2 are safe from this constraint
                pass
        
        # Better approach: for each constraint, determine which colors are safe
        # A color is safe for a constraint if it equals at least one of the other 3 colors
        # OR if the other 3 colors are not all distinct
        
        # Let's compute the set of colors that are safe for ALL constraints
        # Start with all colors as potentially safe
        # For each constraint, intersect with the safe colors for that constraint
        
        # Actually, let's just check each candidate color
        # But optimize by early termination
        
        valid_colors = []
        for c in range(max_color + 1):
            if c == old_color:
                continue
            valid = True
            for others in constraints:
                c0 = colors[others[0]]
                c1 = colors[others[1]]
                c2 = colors[others[2]]
                # If the 3 other colors are all distinct and none equals c, then c would create a rainbow
                if c0 != c1 and c1 != c2 and c0 != c2:
                    if c != c0 and c != c1 and c != c2:
                        valid = False
                        break
            if valid:
                valid_colors.append(c)
        
        if not valid_colors:
            continue
        
        # Prefer colors not in used_colors (to maximize diversity)
        new_used = [c for c in valid_colors if color_counts[c] == 0]
        if new_used:
            # Pick a random new color
            new_color = random.choice(new_used)
        else:
            # Pick a random valid color
            new_color = random.choice(valid_colors)
        
        # Apply the move
        colors[e] = new_color
        color_counts[old_color] -= 1
        color_counts[new_color] += 1
        
        nc = int(np.count_nonzero(color_counts))
        if nc > best_count:
            best = colors.copy()
            best_count = nc
    
    return best.tolist()
# EVOLVE-BLOCK-END
