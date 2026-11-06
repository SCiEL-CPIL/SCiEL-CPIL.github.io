import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Precompute edge index mapping
EDGE_TO_IDX = {}
IDX_TO_EDGE = []
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        EDGE_TO_IDX[(i, j)] = idx
        IDX_TO_EDGE.append((i, j))
        idx += 1

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return EDGE_TO_IDX[(i, j)]

# Precompute all 4-cycles
# There are C(16,4) = 1820 subsets of 4 vertices.
# For each subset {a,b,c,d}, there are 3 cycles:
# 1. a-b-c-d-a
# 2. a-b-d-c-a
# 3. a-c-b-d-a

ALL_CYCLES = []
CYCLES_BY_EDGE = [[] for _ in range(NUM_EDGES)]

for quad in itertools.combinations(range(N), 4):
    a, b, c, d = quad
    # Cycle 1: a-b-c-d-a
    c1_edges = [edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)]
    c1_id = len(ALL_CYCLES)
    ALL_CYCLES.append(c1_edges)
    for e in c1_edges:
        CYCLES_BY_EDGE[e].append(c1_id)
        
    # Cycle 2: a-b-d-c-a
    c2_edges = [edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)]
    c2_id = len(ALL_CYCLES)
    ALL_CYCLES.append(c2_edges)
    for e in c2_edges:
        CYCLES_BY_EDGE[e].append(c2_id)
        
    # Cycle 3: a-c-b-d-a
    c3_edges = [edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)]
    c3_id = len(ALL_CYCLES)
    ALL_CYCLES.append(c3_edges)
    for e in c3_edges:
        CYCLES_BY_EDGE[e].append(c3_id)

def is_valid_local(colors, edge_id):
    """Check if the current coloring is valid with respect to all cycles containing edge_id."""
    for cid in CYCLES_BY_EDGE[edge_id]:
        edges = ALL_CYCLES[cid]
        # Check if 4 edges have distinct colors
        c0 = colors[edges[0]]
        c1 = colors[edges[1]]
        c2 = colors[edges[2]]
        c3 = colors[edges[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    best_colors = [0] * NUM_EDGES
    best_count = 1  # At least color 0 is used
    
    # We will perform several restarts with different random seeds
    # Each restart tries to improve the number of distinct colors
    
    for seed in range(1000):
        if time.time() > deadline:
            break
            
        random.seed(seed)
        colors = [0] * NUM_EDGES
        used_colors = {0}
        
        # Heuristic: Try to assign new colors to random edges
        # We iterate multiple times, trying to "upgrade" edges to new colors
        
        # To speed up, we can track which colors are available.
        # But 120 edges, 120 colors. We can just pick random new colors.
        
        for _ in range(200): # Passes
            if time.time() > deadline:
                break
                
            # Shuffle edge order for diversity
            edge_order = list(range(NUM_EDGES))
            random.shuffle(edge_order)
            
            for eidx in edge_order:
                if time.time() > deadline:
                    break
                    
                # Try to assign a new color not yet used, or a random one
                # Prefer new colors to maximize count
                if len(used_colors) < 120:
                    # Pick a random color not in used_colors
                    # Efficient way: pick random 0-119, if in used, retry a few times or just pick random
                    # Since 120 is small, we can just generate a random one and check
                    new_color = random.randint(0, 119)
                    if new_color in used_colors:
                        # Try a few times to get a new one
                        for _ in range(5):
                            new_color = random.randint(0, 119)
                            if new_color not in used_colors:
                                break
                        if new_color in used_colors:
                            continue # Skip if we can't easily find a new one, or just use existing?
                            # Actually, using existing colors doesn't increase count.
                            # But it might allow *other* edges to take new colors later?
                            # For now, focus on adding new colors.
                            
                old_color = colors[eidx]
                
                # Temporarily apply new color
                colors[eidx] = new_color
                
                # Check validity locally
                if is_valid_local(colors, eidx):
                    used_colors.add(new_color)
                    # Accept
                else:
                    # Revert
                    colors[eidx] = old_color
        
        cur_count = len(used_colors)
        if cur_count > best_count:
            best_count = cur_count
            best_colors = colors[:]
            
    return best_colors
