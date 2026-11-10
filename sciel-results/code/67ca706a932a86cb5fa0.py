"""
Edge coloring of K_16 with no rainbow 4-cycle.
Strategy:
  1. Greedy construction: assign colors one edge at a time, trying to maximize
     the number of distinct colors while avoiding any 4-cycle with 4 distinct colors.
  2. Local search: try to change colors of edges to introduce new colors,
     using an efficient local validity check (only cycles containing the modified edge).
  3. Deadline guard: return best result before 90s wall-clock.
"""
import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
DEADLINE = 85.0  # safety margin

# Precompute edge indices
EDGE_INDEX = {}
for i in range(N):
    for j in range(i + 1, N):
        EDGE_INDEX[(i, j)] = i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

# Precompute for each edge, which 4-cycles contain it.
# A 4-cycle on vertices {a,b,c,d} (sorted) has edges:
#   (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
# The three 4-cycles (cycles of length 4) on a 4-set {a,b,c,d} are:
#   (a,b,c,d), (a,b,d,c), (a,c,b,d)
# But actually, a "four-cycle" in a 4-vertex complete graph K4:
# There are 3 distinct 4-cycles (up to rotation/reversal) in K4:
#   1: a-b-c-d-a  (edges: ab, bc, cd, da)
#   2: a-b-d-c-a  (edges: ab, bd, dc, ca)
#   3: a-c-b-d-a  (edges: ac, cb, bd, da)
# Each uses 4 edges.

# For each edge e, list all (cycle_id, edge_positions) where the cycle is a 4-cycle
# containing e. We'll store for each edge a list of "conflict groups":
# Each conflict group is a set of 4 edges forming a 4-cycle.

# Build all 4-cycles
ALL_CYCLES = []
CYCLE_EDGE_SETS = []  # list of frozensets of 4 edge indices

for quad in itertools.combinations(range(N), 4):
    a, b, c, d = quad
    # Three 4-cycles in K4:
    cycles = [
        [(a,b), (b,c), (c,d), (d,a)],
        [(a,b), (b,d), (d,c), (c,a)],
        [(a,c), (c,b), (b,d), (d,a)],
    ]
    for cyc in cycles:
        edge_set = frozenset(EDGE_INDEX[e] for e in cyc)
        CYCLE_EDGE_SETS.append(edge_set)
        ALL_CYCLES.append(cyc)

# For each edge, list of cycle indices that contain it
EDGE_TO_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, es in enumerate(CYCLE_EDGE_SETS):
    for e in es:
        EDGE_TO_CYCLES[e].append(ci)

def count_colors(colors):
    return len(set(colors))

def is_rainbow_cycle(colors, cycle_edges):
    """Check if the 4 edges in cycle_edges have 4 distinct colors."""
    cs = [colors[e] for e in cycle_edges]
    return len(set(cs)) == 4

def check_local_validity(colors, edge_idx):
    """Check if changing edge_idx's color would create a rainbow 4-cycle.
    Only checks cycles containing edge_idx."""
    for ci in EDGE_TO_CYCLES[edge_idx]:
        es = CYCLE_EDGE_SETS[ci]
        if is_rainbow_cycle(colors, es):
            return False
    return True

def is_fully_valid(colors):
    """Full validity check."""
    for es in CYCLE_EDGE_SETS:
        if is_rainbow_cycle(colors, es):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Greedy construction
    colors = [0] * NUM_EDGES
    used_colors = {0}
    current_count = 1
    
    # Order edges: try to assign colors in a structured way
    # Use a heuristic: process edges in an order that spreads colors
    edge_order = list(range(NUM_EDGES))
    random.seed(42)
    random.shuffle(edge_order)
    
    # Greedy: for each edge, try to assign a new color first, then existing colors
    for ei in edge_order:
        if time.time() > deadline - 10:
            break
        # Try new colors first (to maximize distinct count)
        assigned = False
        for c in range(120):
            if c not in used_colors:
                colors[ei] = c
                if check_local_validity(colors, ei):
                    used_colors.add(c)
                    current_count += 1
                    assigned = True
                    break
        if not assigned:
            # Try existing colors
            for c in sorted(used_colors):
                colors[ei] = c
                if check_local_validity(colors, ei):
                    assigned = True
                    break
        if not assigned:
            colors[ei] = 0  # fallback
        
        # Check deadline
        if time.time() > deadline - 15:
            break
    
    # Ensure full validity (greedy local checks should ensure this, but verify)
    if not is_fully_valid(colors):
        # Fallback: fix by resetting problematic edges
        for es in CYCLE_EDGE_SETS:
            if is_rainbow_cycle(colors, es):
                # Change one edge in the cycle to match another
                for e in es:
                    for other_e in es:
                        if other_e != e:
                            colors[e] = colors[other_e]
                            break
                    break
    
    best_count = count_colors(colors)
    best_colors = list(colors)
    
    # Local search: try to improve by changing colors
    # Strategy: pick an edge, try changing its color to a new unused color
    # or to a different existing color, if it doesn't create a rainbow cycle
    # and increases (or maintains) the distinct color count.
    
    rng = random.Random(123)
    
    while time.time() < deadline - 5:
        # Pick a random edge
        ei = rng.randrange(NUM_EDGES)
        old_color = colors[ei]
        
        # Try to change to a new color not in used_colors
        # Find unused colors
        used_set = set(colors)
        unused = [c for c in range(120) if c not in used_set]
        
        # Try unused colors first
        rng.shuffle(unused)
        improved = False
        for c in unused:
            colors[ei] = c
            if check_local_validity(colors, ei):
                # Check if this actually increases distinct count
                # It does if c was not used before
                improved = True
                break
        if not improved:
            # Try existing colors (might allow other edges to get new colors later)
            existing = list(used_set)
            rng.shuffle(existing)
            for c in existing:
                if c == old_color:
                    continue
                colors[ei] = c
                if check_local_validity(colors, ei):
                    improved = True
                    break
        if not improved:
            colors[ei] = old_color
        else:
            new_count = count_colors(colors)
            if new_count > best_count:
                best_count = new_count
                best_colors = list(colors)
    
    # Final full validity check
    if not is_fully_valid(best_colors):
        # Use the greedy result which should be valid
        return colors
    
    return best_colors

# Ensure the function is accessible
if __name__ == "__main__":
    result = construct()
    print(f"Colors used: {len(set(result))}")
