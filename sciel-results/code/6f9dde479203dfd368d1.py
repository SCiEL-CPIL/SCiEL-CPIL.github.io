"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Precompute edge indices
def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2

# Build edge list in lexicographic order
EDGE_LIST = []
for i in range(N):
    for j in range(i + 1, N):
        EDGE_LIST.append((i, j))

EDGE_TO_IDX = {}
for idx, (i, j) in enumerate(EDGE_LIST):
    EDGE_TO_IDX[(i, j)] = idx

# Precompute all 4-cycles: for each set of 4 vertices, the 3 possible 4-cycles
# A 4-cycle on {a,b,c,d} can be: a-b-c-d-a, a-b-d-c-a, a-c-b-d-a
# Each 4-cycle is defined by 4 edges
FOUR_CYCLES = []  # list of tuples of 4 edge indices
for quad in itertools.combinations(range(N), 4):
    a, b, c, d = quad
    # Three 4-cycles:
    # 1: a-b, b-c, c-d, d-a
    # 2: a-b, b-d, d-c, c-a
    # 3: a-c, c-b, b-d, d-a
    cycles = [
        [(a,b), (b,c), (c,d), (d,a)],
        [(a,b), (b,d), (d,c), (c,a)],
        [(a,c), (c,b), (b,d), (d,a)],
    ]
    for cyc in cycles:
        edge_idxs = []
        for (u, v) in cyc:
            if u > v:
                u, v = v, u
            edge_idxs.append(EDGE_TO_IDX[(u, v)])
        FOUR_CYCLES.append(tuple(edge_idxs))

# For each edge, which 4-cycles does it belong to?
EDGE_TO_CYCLES = [[] for _ in range(NUM_EDGES)]
for cyc_idx, cyc in enumerate(FOUR_CYCLES):
    for e in cyc:
        EDGE_TO_CYCLES[e].append(cyc_idx)

def is_valid(colors):
    """Check if the coloring is valid (no 4-cycle has 4 distinct colors)."""
    for cyc in FOUR_CYCLES:
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def count_colors(colors):
    return len(set(colors))

def try_recolor(colors, edge_idx, new_color):
    """Try to set colors[edge_idx] = new_color, return True if valid."""
    old_color = colors[edge_idx]
    if old_color == new_color:
        return True
    colors[edge_idx] = new_color
    # Check all 4-cycles containing this edge
    for cyc_idx in EDGE_TO_CYCLES[edge_idx]:
        cyc = FOUR_CYCLES[cyc_idx]
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            colors[edge_idx] = old_color
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Initialize with all zeros
    colors = [0] * NUM_EDGES
    best_colors = colors[:]
    best_count = 1
    
    # Try to introduce new colors via local search
    random.seed(42)
    
    # First pass: greedy - try to assign smallest possible new color to each edge
    for e in range(NUM_EDGES):
        for new_c in range(1, 120):
            if try_recolor(colors, e, new_c):
                break
            # revert
            colors[e] = 0
    
    # Check validity
    if is_valid(colors):
        cnt = count_colors(colors)
        if cnt > best_count:
            best_count = cnt
            best_colors = colors[:]
    
    # Local search: try to improve
    for iteration in range(500000):
        if time.time() > deadline:
            break
        
        # Pick a random edge
        e = random.randint(0, NUM_EDGES - 1)
        
        # Try to change to a random new color (prefer unused colors)
        used_colors = set(best_colors)
        unused = [c for c in range(120) if c not in used_colors]
        
        if unused:
            candidates = unused[:10] + [best_colors[e]]
        else:
            candidates = [best_colors[e]] + random.sample(range(120), min(10, 120))
        
        random.shuffle(candidates)
        
        improved = False
        for new_c in candidates:
            if new_c == best_colors[e]:
                continue
            # Try on a copy
            test_colors = best_colors[:]
            if try_recolor(test_colors, e, new_c):
                new_cnt = count_colors(test_colors)
                if new_cnt > best_count:
                    best_colors = test_colors
                    best_count = new_cnt
                    improved = True
                    break
                elif new_cnt == best_count:
                    # Keep it if same count (for diversity in search)
                    best_colors = test_colors
                    improved = True
                    break
    
    # Final verification
    if not is_valid(best_colors):
        # Fallback to all zeros
        return [0] * NUM_EDGES
    
    return best_colors

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
