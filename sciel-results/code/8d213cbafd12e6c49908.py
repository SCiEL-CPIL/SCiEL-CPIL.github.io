"""
Edge coloring of K16 to maximize distinct colors subject to:
  No 4-cycle has 4 distinct edge colors.

Approach:
  - Precompute all 4-cycles (three per 4-vertex subset).
  - Greedy construction: assign colors to edges in a fixed order, picking the
    smallest unused color that doesn't violate any 4-cycle constraint.
  - Then local search (swap / recolor) to try to increase the number of distinct
    colors used, while maintaining validity.
  - Deadline guard: return best valid coloring before 85 s.
"""
import time
import random
from itertools import combinations

N = 16
E = N * (N - 1) // 2  # 120

# Precompute edge indices
def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + j - i - 1

# Build the list of all 4-cycles.
# A 4-cycle on vertices {a,b,c,d} (sorted) has three possible pairings:
#   (a,b,c,d), (a,b,d,c), (a,c,b,d)  -- actually the three Hamiltonian 4-cycles.
# More precisely, for a 4-set {v0,v1,v2,v3}, the three 4-cycles are:
#   v0-v1-v2-v3-v0
#   v0-v1-v3-v2-v0
#   v0-v2-v1-v3-v0
# Each cycle uses 4 edges.

def build_four_cycles():
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        cycles.append([(a,b),(b,c),(c,d),(d,a)])
        # Cycle 2: a-b-d-c-a
        cycles.append([(a,b),(b,d),(d,c),(c,a)])
        # Cycle 3: a-c-b-d-a
        cycles.append([(a,c),(c,b),(b,d),(d,a)])
    return cycles

FOUR_CYCLES = build_four_cycles()

# For fast checking, map each edge to the list of 4-cycles it participates in.
edge_to_cycles = [[] for _ in range(E)]
for ci, cyc in enumerate(FOUR_CYCLES):
    for (u, v) in cyc:
        ei = edge_index(u, v)
        edge_to_cycles[ei].append(ci)

def count_distinct_colors(coloring):
    return len(set(coloring))

def is_valid(coloring):
    """Check that no 4-cycle has 4 distinct edge colors."""
    for cyc in FOUR_CYCLES:
        colors = [coloring[edge_index(u, v)] for (u, v) in cyc]
        if len(set(colors)) == 4:
            return False
    return True

def check_cycle_valid(coloring, ci):
    """Check a single 4-cycle."""
    cyc = FOUR_CYCLES[ci]
    colors = [coloring[edge_index(u, v)] for (u, v) in cyc]
    return len(set(colors)) < 4

def is_valid_change(coloring, edge_i, new_color):
    """Check if setting coloring[edge_i] = new_color keeps all cycles valid.
    Only need to check cycles containing edge_i."""
    old_color = coloring[edge_i]
    if old_color == new_color:
        return True
    for ci in edge_to_cycles[edge_i]:
        cyc = FOUR_CYCLES[ci]
        colors = []
        for (u, v) in cyc:
            ei = edge_index(u, v)
            if ei == edge_i:
                colors.append(new_color)
            else:
                colors.append(coloring[ei])
        if len(set(colors)) == 4:
            return False
    return True

def greedy_construct():
    """Greedy: assign smallest valid unused color to each edge in order."""
    coloring = [0] * E
    used_colors = set()
    for ei in range(E):
        # Try to use a new color first (to maximize distinct count)
        # Then fall back to existing colors
        # Actually, to maximize distinct colors, we want to use a NEW color if possible.
        # But we also need to be valid. Let's try colors in order of preference:
        # Prefer new colors (highest unused) to maximize distinct count.
        # Actually, let's just try all colors 0..119 and pick the first valid one,
        # but prefer new colors.
        
        # Strategy: try to assign a color that hasn't been used yet, if valid.
        # If no new color works, use an existing valid color.
        assigned = False
        # Try new colors (from 119 down to 0, but only those not in used_colors)
        for c in range(120):
            if c in used_colors:
                continue
            if is_valid_change(coloring, ei, c):
                coloring[ei] = c
                used_colors.add(c)
                assigned = True
                break
        if not assigned:
            # Fall back to existing colors
            for c in sorted(used_colors):
                if is_valid_change(coloring, ei, c):
                    coloring[ei] = c
                    assigned = True
                    break
        if not assigned:
            # Shouldn't happen; use 0
            coloring[ei] = 0
    return coloring

def local_search(coloring, deadline):
    """Try to increase distinct colors via recoloring."""
    best = coloring[:]
    best_count = count_distinct_colors(best)
    rng = random.Random(42)
    
    while time.time() < deadline:
        improved = False
        # Try to recolor edges to introduce new colors
        edges = list(range(E))
        rng.shuffle(edges)
        
        for ei in edges:
            if time.time() > deadline:
                break
            current_color = best[ei]
            # Try to change to a new color
            for c in range(120):
                if c == current_color:
                    continue
                if c in set(best):
                    continue  # only try new colors
                if is_valid_change(best, ei, c):
                    best[ei] = c
                    new_count = count_distinct_colors(best)
                    if new_count > best_count:
                        best_count = new_count
                        improved = True
                    break  # only one change per edge per pass
        
        if not improved:
            # Try swapping colors between two edges to free up a color slot
            # Actually, let's try: pick an edge with a "common" color and try to
            # recolor it to a rare/new color.
            break
    
    return best

def construct():
    deadline = time.time() + 85  # safety margin
    
    # Phase 1: Greedy construction
    coloring = greedy_construct()
    if not is_valid(coloring):
        # Fallback: all zeros
        return [0] * E
    
    initial_count = count_distinct_colors(coloring)
    
    # Phase 2: Local search to improve
    if time.time() < deadline:
        coloring = local_search(coloring, deadline)
    
    final_count = count_distinct_colors(coloring)
    
    # Final validation
    if not is_valid(coloring):
        return [0] * E
    
    return coloring
