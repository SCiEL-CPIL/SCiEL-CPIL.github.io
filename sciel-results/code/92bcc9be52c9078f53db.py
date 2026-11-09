import random
import time
import itertools
import numpy as np

N = 20
C = 20

# Precompute all 4-cycles in K_N
# A 4-cycle is a set of 4 vertices {u,v,w,x} with edges (u,v), (v,w), (w,x), (x,u)
# In a complete graph, any 4 vertices form 3 distinct 4-cycles (up to rotation/reflection).
# The cycles are:
# 1. u-v-w-x-u
# 2. u-v-x-w-u
# 3. u-w-v-x-u

def generate_cycles(n):
    cycles = []
    for combo in itertools.combinations(range(n), 4):
        u, v, w, x = combo
        # 3 distinct cycles on these 4 vertices
        cycles.append((u, v, w, x))
        cycles.append((u, v, x, w))
        cycles.append((u, w, v, x))
    return cycles

CYCLES = generate_cycles(N)
NUM_CYCLES = len(CYCLES)

# Precompute for each edge (i,j), which cycles it participates in and its position in the cycle
# edge_to_cycles[edge_index] = list of (cycle_index, position)
# where position 0 means edge (u,v), 1 means (v,w), 2 means (w,x), 3 means (x,u)
EDGE_TO_CYCLES = {}
for ci, (u, v, w, x) in enumerate(CYCLES):
    edges_in_cycle = [(u, v), (v, w), (w, x), (x, u)]
    for pos, (a, b) in enumerate(edges_in_cycle):
        key = (min(a, b), max(a, b))
        if key not in EDGE_TO_CYCLES:
            EDGE_TO_CYCLES[key] = []
        EDGE_TO_CYCLES[key].append((ci, pos))

# Convert to numpy arrays for fast lookup
# We'll use a 2D array: cycle_edge_positions[cycle_idx, edge_idx_in_cycle] = edge_id
# But it's easier to work with the list structure.

# For faster checking, let's precompute for each cycle the 4 edge keys
CYCLE_EDGE_KEYS = []
for (u, v, w, x) in CYCLES:
    keys = [(min(u,v), max(u,v)), (min(v,w), max(v,w)), (min(w,x), max(w,x)), (min(x,u), max(x,u))]
    CYCLE_EDGE_KEYS.append(keys)

# Precompute for each edge, the list of (cycle_idx, pos)
# This is already in EDGE_TO_CYCLES

def check_rainbow(col):
    """
    col: 2D numpy array of shape (N, N). -1 means no edge.
    Returns True if all 4-cycles that are fully present are rainbow.
    """
    for ci in range(NUM_CYCLES):
        keys = CYCLE_EDGE_KEYS[ci]
        colors = []
        all_present = True
        for (a, b) in keys:
            c = col[a, b]
            if c == -1:
                all_present = False
                break
            colors.append(c)
        if not all_present:
            continue
        # Check if all 4 colors are distinct
        if len(set(colors)) < 4:
            return False
    return True

def get_invalid_cycles(col):
    """
    Returns list of (cycle_idx, [c1, c2, c3, c4]) for invalid cycles.
    """
    invalid = []
    for ci in range(NUM_CYCLES):
        keys = CYCLE_EDGE_KEYS[ci]
        colors = []
        all_present = True
        for (a, b) in keys:
            c = col[a, b]
            if c == -1:
                all_present = False
                break
            colors.append(c)
        if not all_present:
            continue
        if len(set(colors)) < 4:
            invalid.append((ci, colors))
    return invalid

def _greedy_construct(seed):
    rng = random.Random(seed)
    
    # Initialize with complete graph and random colors
    col = np.full((N, N), -1, dtype=np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            c = rng.randint(0, C - 1)
            col[i, j] = c
            col[j, i] = c
            
    # Phase 1: Repair - remove edges to make valid
    max_repair_steps = 20000
    steps = 0
    while steps < max_repair_steps:
        invalid = get_invalid_cycles(col)
        if not invalid:
            break
        
        # Pick a random invalid cycle
        ci, colors = rng.choice(invalid)
        keys = CYCLE_EDGE_KEYS[ci]
        
        # Try to recolor an edge in the cycle
        success = False
        # Try each edge in the cycle, each with random colors
        for pos in range(4):
            a, b = keys[pos]
            old_c = col[a, b]
            # Try all colors
            for new_c in range(C):
                if new_c == old_c:
                    continue
                col[a, b] = new_c
                col[b, a] = new_c
                if check_rainbow(col):
                    success = True
                    break
                else:
                    col[a, b] = old_c
                    col[b, a] = old_c
            if success:
                break
        
        if not success:
            # Remove an edge from the cycle
            pos = rng.randint(0, 3)
            a, b = keys[pos]
            col[a, b] = -1
            col[b, a] = -1
        
        steps += 1
    
    if get_invalid_cycles(col):
        return []
    
    # Phase 2: Grow - add missing edges
    missing_edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if col[i, j] == -1:
                missing_edges.append((i, j))
                
    rng.shuffle(missing_edges)
    
    for u, v in missing_edges:
        found_color = False
        colors = list(range(C))
        rng.shuffle(colors)
        for c in colors:
            col[u, v] = c
            col[v, u] = c
            if check_rainbow(col):
                found_color = True
                break
            else:
                col[u, v] = -1
                col[v, u] = -1
        
    # Convert to output format
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if col[i, j] != -1:
                edges.append([i, j, int(col[i, j])])
                
    return edges

def _edge_ok_fast(col, u, v, c):
    """
    Check if adding edge (u,v) with color c to the current graph keeps all 4-cycles rainbow.
    Only need to check cycles that include edge (u,v).
    """
    key = (min(u, v), max(u, v))
    if key not in EDGE_TO_CYCLES:
        return True
    
    for ci, pos in EDGE_TO_CYCLES[key]:
        keys = CYCLE_EDGE_KEYS[ci]
        colors = []
        all_present = True
        for p, (a, b) in enumerate(keys):
            if p == pos:
                colors.append(c)
            else:
                cc = col[a, b]
                if cc == -1:
                    all_present = False
                    break
                colors.append(cc)
        if not all_present:
            continue
        if len(set(colors)) < 4:
            return False
    return True

def _greedy_construct_v2(seed):
    rng = random.Random(seed)
    
    # Start with empty graph
    col = np.full((N, N), -1, dtype=np.int32)
    
    # Try to add all edges in random order
    all_edges = []
    for i in range(N):
        for j in range(i + 1, N):
            all_edges.append((i, j))
    rng.shuffle(all_edges)
    
    for u, v in all_edges:
        colors = list(range(C))
        rng.shuffle(colors)
        for c in colors:
            if _edge_ok_fast(col, u, v, c):
                col[u, v] = c
                col[v, u] = c
                break
    
    # Convert to output format
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if col[i, j] != -1:
                edges.append([i, j, int(col[i, j])])
                
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_edges = []
    max_edges = 0
    
    # Try multiple strategies
    # Strategy 1: Greedy from empty (v2)
    seed = 42
    while time.time() < deadline:
        edges = _greedy_construct_v2(seed)
        if len(edges) > max_edges:
            max_edges = len(edges)
            best_edges = edges
        seed += 1
        if seed % 50 == 0 and time.time() > deadline - 30:
            # Switch to repair+grow strategy
            break
    
    # Strategy 2: Repair and grow
    while time.time() < deadline - 5:
        edges = _greedy_construct(seed)
        if len(edges) > max_edges:
            max_edges = len(edges)
            best_edges = edges
        seed += 1
    
    return best_edges
