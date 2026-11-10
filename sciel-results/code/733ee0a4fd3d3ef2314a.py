import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Precompute edge index
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

# Precompute all 4-cycles
def build_cycles():
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # 3 cycles on 4 vertices
        # Cycle 1: a-b-c-d-a
        e1 = [edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)]
        cycles.append(e1)
        # Cycle 2: a-b-d-c-a
        e2 = [edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)]
        cycles.append(e2)
        # Cycle 3: a-c-b-d-a
        e3 = [edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)]
        cycles.append(e3)
    return cycles

# Precompute edge -> list of cycle indices
def build_edge_to_cycles(cycles):
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(cycles):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    return edge_to_cycles

_CYCLES = build_cycles()
_EDGE_TO_CYCLES = build_edge_to_cycles(_CYCLES)

def check_valid(colors):
    """Check if coloring is valid."""
    for cyc in _CYCLES:
        if len(set(colors[e] for e in cyc)) == 4:
            return False
    return True

def count_distinct(colors):
    return len(set(colors))

def greedy_construct(seed=42):
    random.seed(seed)
    colors = [0] * NUM_EDGES
    used_colors = {0}
    
    # Try to assign new colors greedily
    for e in range(NUM_EDGES):
        # Try colors in order: prefer new colors
        best_color = 0
        for c in range(120):
            # Check if using color c for edge e violates any cycle
            valid = True
            for ci in _EDGE_TO_CYCLES[e]:
                cyc = _CYCLES[ci]
                # Build the set of colors in this cycle with e=c
                col_set = set()
                for ee in cyc:
                    if ee == e:
                        col_set.add(c)
                    else:
                        col_set.add(colors[ee])
                    if len(col_set) > 3:
                        valid = False
                        break
            if valid:
                best_color = c
                break
        colors[e] = best_color
        used_colors.add(best_color)
    
    return colors

def local_search(colors, max_time=50):
    start_time = time.time()
    colors = list(colors)
    best_distinct = count_distinct(colors)
    
    while time.time() - start_time < max_time:
        improved = False
        # Try to change each edge to a new color
        for e in range(NUM_EDGES):
            old_color = colors[e]
            # Try to find a new color
            for c in range(120):
                if c == old_color:
                    continue
                # Check if changing e to c is valid
                valid = True
                for ci in _EDGE_TO_CYCLES[e]:
                    cyc = _CYCLES[ci]
                    col_set = set()
                    for ee in cyc:
                        if ee == e:
                            col_set.add(c)
                        else:
                            col_set.add(colors[ee])
                        if len(col_set) > 3:
                            valid = False
                            break
                if valid:
                    colors[e] = c
                    new_distinct = count_distinct(colors)
                    if new_distinct > best_distinct:
                        best_distinct = new_distinct
                        improved = True
                    break  # Take the first valid new color
        if not improved:
            break
    
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Try multiple greedy seeds
    best_colors = None
    best_score = 0
    
    for seed in range(20):
        if time.time() > deadline - 5:
            break
        colors = greedy_construct(seed)
        score = count_distinct(colors)
        if score > best_score:
            best_score = score
            best_colors = colors
    
    # Local search on the best
    if best_colors is not None and time.time() < deadline - 5:
        best_colors = local_search(best_colors, max_time=deadline - time.time() - 2)
    
    return best_colors if best_colors is not None else [0] * NUM_EDGES
