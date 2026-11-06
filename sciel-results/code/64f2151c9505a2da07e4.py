"""Rewritten program: backtracking with MRV + new-color-first heuristic for max distinct colors in C4-free edge coloring of K16."""
import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Precompute all 4-cycles (as edge index sets)
# A 4-cycle on vertices a<b<c<d has edges: (a,b),(b,c),(c,d),(d,a)
# The three 4-cycles on {a,b,c,d} are:
#   C1: a-b-c-d-a  -> edges (a,b),(b,c),(c,d),(d,a)
#   C2: a-b-d-c-a  -> edges (a,b),(b,d),(d,c),(c,a)
#   C3: a-c-b-d-a  -> edges (a,c),(c,b),(b,d),(d,a)

def _edge_idx(i, j):
    """Map pair (i,j) with i<j to edge index 0..119 in lex order."""
    # Lex order: (0,1),(0,2),...,(0,15),(1,2),...,(14,15)
    # Index = sum_{k=0}^{i-1} (N-1-k) + (j - i - 1)
    # = i*(N-1) - i*(i+1)//2 + j - i - 1
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

# Build list of all 4-cycles as tuples of 4 edge indices
ALL_CYCLES = []
# For each set of 4 vertices {a,b,c,d} with a<b<c<d, the three 4-cycles:
# Vertices in order a,b,c,d
# Cycle 1: a-b, b-c, c-d, d-a
# Cycle 2: a-b, b-d, d-c, c-a  (i.e., a-b, b-d, c-d, a-c)
# Cycle 3: a-c, c-b, b-d, d-a  (i.e., a-c, b-c, b-d, a-d)

for a, b, c, d in combinations(range(N), 4):
    # Cycle 1: edges (a,b), (b,c), (c,d), (d,a)
    e1 = (_edge_idx(a, b), _edge_idx(b, c), _edge_idx(c, d), _edge_idx(a, d))
    ALL_CYCLES.append(e1)
    # Cycle 2: edges (a,b), (b,d), (c,d), (a,c)
    e2 = (_edge_idx(a, b), _edge_idx(b, d), _edge_idx(c, d), _edge_idx(a, c))
    ALL_CYCLES.append(e2)
    # Cycle 3: edges (a,c), (b,c), (b,d), (a,d)
    e3 = (_edge_idx(a, c), _edge_idx(b, c), _edge_idx(b, d), _edge_idx(a, d))
    ALL_CYCLES.append(e3)

NUM_CYCLES = len(ALL_CYCLES)

# For each edge, which cycles contain it
EDGE_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cyc in enumerate(ALL_CYCLES):
    for e in cyc:
        EDGE_CYCLES[e].append(ci)


def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    best_colors = [0] * NUM_EDGES
    best_count = 1  # at least 1 color used

    def check_cycle(cyc, colors):
        """Return True if the 4 edges in cyc have 4 distinct colors (violation)."""
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        return len({c0, c1, c2, c3}) == 4

    def is_valid_assignment(colors, edge_idx, color):
        """Check if assigning 'color' to edge_idx violates any 4-cycle constraint."""
        # Only need to check cycles containing this edge
        for ci in EDGE_CYCLES[edge_idx]:
            cyc = ALL_CYCLES[ci]
            # Check if all 4 edges in this cycle are assigned
            if colors[cyc[0]] == -1 or colors[cyc[1]] == -1 or colors[cyc[2]] == -1 or colors[cyc[3]] == -1:
                continue
            # Temporarily assign
            old = colors[edge_idx]
            colors[edge_idx] = color
            if check_cycle(cyc, colors):
                colors[edge_idx] = old
                return False
            colors[edge_idx] = old
        return True

    def count_distinct(colors):
        return len(set(colors))

    # Backtracking with MRV heuristic
    # We'll assign edges in a smart order. Start with all -1 (unassigned).
    # Use iterative deepening on number of distinct colors? No, just greedy backtracking.

    # Strategy: assign edges one by one. At each step, pick the unassigned edge
    # that has the most constraints (MRV: fewest available colors, or most cycles).
    # Prefer trying a new color first to maximize diversity.

    def backtrack(colors, assigned_count, used_colors, deadline):
        nonlocal best_colors, best_count
        if time.time() > deadline:
            return

        if assigned_count == NUM_EDGES:
            cnt = count_distinct(colors)
            if cnt > best_count:
                best_count = cnt
                best_colors = colors[:]
            return

        # Find unassigned edge with MRV (most constrained)
        # For simplicity, pick the unassigned edge that appears in the most already-partially-assigned cycles
        # Actually, let's use a simpler heuristic: pick the edge with the most cycles where 3 other edges are already assigned
        best_edge = -1
        best_score = -1
        for e in range(NUM_EDGES):
            if colors[e] != -1:
                continue
            # Count how many cycles containing e have 3 other edges assigned
            score = 0
            for ci in EDGE_CYCLES[e]:
                cyc = ALL_CYCLES[ci]
                other_assigned = 0
                for oe in cyc:
                    if oe != e and colors[oe] != -1:
                        other_assigned += 1
                if other_assigned == 3:
                    score += 2  # very constrained
                elif other_assigned == 2:
                    score += 1
            if score > best_score:
                best_score = score
                best_edge = e
                if score >= 3:  # good enough
                    break

        if best_edge == -1:
            # No unassigned edges found? Shouldn't happen
            return

        # Try colors: prefer new colors first (to maximize distinct count)
        # Then try existing colors
        # New colors: we can use any color from 0 to 119 not yet used
        # But we only need to consider up to some max. Let's try new colors up to best_count+5 or so
        # Actually, for maximizing distinct colors, try a new color first.

        # Determine which new colors to try
        new_color_candidates = []
        for c in range(best_count + 5):  # try colors up to a reasonable bound
            if c not in used_colors:
                new_color_candidates.append(c)

        # Try new colors first
        for c in new_color_candidates:
            if time.time() > deadline:
                return
            if is_valid_assignment(colors, best_edge, c):
                colors[best_edge] = c
                used_colors.add(c)
                backtrack(colors, assigned_count + 1, used_colors, deadline)
                used_colors.discard(c)
                colors[best_edge] = -1

        # Try existing colors
        for c in sorted(used_colors):
            if time.time() > deadline:
                return
            if is_valid_assignment(colors, best_edge, c):
                colors[best_edge] = c
                backtrack(colors, assigned_count + 1, used_colors, deadline)
                colors[best_edge] = -1

    # Initialize
    colors = [-1] * NUM_EDGES
    used_colors = set()
    backtrack(colors, 0, used_colors, deadline)

    # If we didn't find anything better than 1, at least return something valid
    if best_count <= 1:
        # Try a simple greedy
        colors = [0] * NUM_EDGES
        for e in range(NUM_EDGES):
            for c in range(120):
                if is_valid_assignment(colors, e, c):
                    colors[e] = c
                    break
        cnt = count_distinct(colors)
        if cnt > best_count:
            best_count = cnt
            best_colors = colors[:]

    return best_colors
