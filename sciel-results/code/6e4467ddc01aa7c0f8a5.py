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

# Precompute all 4-cycles
FOUR_CYCLES = []
for quad in itertools.combinations(range(N), 4):
    a, b, c, d = quad
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

def find_new_color_candidates(colors):
    """
    Find edges that can be assigned a NEW color (not currently in use).
    Condition: For every 4-cycle containing the edge, the other 3 edges must have <= 2 distinct colors.
    """
    used_colors = set(colors)
    candidates = []
    for e in range(NUM_EDGES):
        is_candidate = True
        for cyc_idx in EDGE_TO_CYCLES[e]:
            cyc = FOUR_CYCLES[cyc_idx]
            other_edges = [x for x in cyc if x != e]
            other_colors = {colors[x] for x in other_edges}
            if len(other_colors) > 2:
                is_candidate = False
                break
        if is_candidate:
            candidates.append(e)
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    best_colors = None
    best_count = 0
    
    # Multiple restarts with different random seeds for initial valid coloring
    for seed in range(100):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        
        # Step 1: Generate a valid base coloring with few colors
        # Use a simple 3-coloring pattern
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                idx = EDGE_TO_IDX[(i, j)]
                # Simple modular coloring, verify later
                colors[idx] = (i + j) % 3
        
        if not is_valid(colors):
            # Fallback to a known valid 3-coloring if the modular one fails
            # Actually, let's just use a valid 3-coloring from a known construction or random search
            # For now, if invalid, try random 3-coloring
            for _ in range(1000):
                colors = [random.randint(0, 2) for _ in range(NUM_EDGES)]
                if is_valid(colors):
                    break
            
            if not is_valid(colors):
                continue

        current_count = count_colors(colors)
        
        # Step 2: Greedy addition of new colors
        # We try to add colors 3, 4, 5, ...
        next_color = 3
        while True:
            if time.time() > deadline:
                break
                
            candidates = find_new_color_candidates(colors)
            if not candidates:
                # No edge can take a new color directly.
                # Try a local search to "open up" a candidate?
                # For simplicity, we just break and restart with a different seed
                break
                
            # Pick a random candidate
            e = random.choice(candidates)
            colors[e] = next_color
            next_color += 1
            current_count += 1
            
            if current_count > best_count:
                best_count = current_count
                best_colors = colors[:]
                
        # Step 3: Local search to try to improve further or unstick
        # If we are stuck, we might try to recolor some edges to existing colors 
        # to potentially allow a new color later, but this is complex.
        # Instead, we rely on the multiple restarts.
        
    # Final verification
    if best_colors is None:
        # Fallback to all zeros if something went wrong
        return [0] * NUM_EDGES
        
    if not is_valid(best_colors):
        return [0] * NUM_EDGES
        
    return best_colors

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
