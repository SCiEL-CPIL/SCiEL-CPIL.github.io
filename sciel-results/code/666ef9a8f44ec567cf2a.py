"""Program for the edge-coloring task: maximize distinct colors in K16 with no rainbow 4-cycle."""
import time
import random
import itertools

N = 16
NUM_COLORS = 120
NUM_EDGES = N * (N - 1) // 2  # 120

def _edge_index(i, j):
    """Get the index for edge (i,j) in lexicographic ordering."""
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + j - i - 1

def _precompute_cycles():
    """Precompute all 4-cycles as lists of 4 edge indices."""
    cycles = []
    for a, b, c, d in itertools.combinations(range(N), 4):
        # Cycle 1: a-b, b-c, c-d, d-a
        cycles.append([_edge_index(a,b), _edge_index(b,c), _edge_index(c,d), _edge_index(d,a)])
        # Cycle 2: a-b, b-d, d-c, c-a
        cycles.append([_edge_index(a,b), _edge_index(b,d), _edge_index(c,d), _edge_index(c,a)])
        # Cycle 3: a-c, c-b, b-d, d-a
        cycles.append([_edge_index(a,c), _edge_index(b,c), _edge_index(b,d), _edge_index(d,a)])
    return cycles

def _precompute_edge_cycles():
    """For each edge, precompute the list of cycles it participates in."""
    all_cycles = _precompute_cycles()
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(all_cycles):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    return all_cycles, edge_to_cycles

def _is_valid_change(colors, edge_idx, new_color, all_cycles, edge_to_cycles):
    """Check if changing edge_idx to new_color keeps all cycles valid."""
    for ci in edge_to_cycles[edge_idx]:
        cyc = all_cycles[ci]
        # Optimization: manually unroll the set creation for speed
        c1 = colors[cyc[0]] if cyc[0] != edge_idx else None
        c2 = colors[cyc[1]] if cyc[1] != edge_idx else None
        c3 = colors[cyc[2]] if cyc[2] != edge_idx else None
        c4 = colors[cyc[3]] if cyc[3] != edge_idx else None
        
        # The other 3 colors
        others = []
        if c1 is not None: others.append(c1)
        if c2 is not None: others.append(c2)
        if c3 is not None: others.append(c3)
        if c4 is not None: others.append(c4)
        
        # Check if the 3 others are distinct and new_color is not among them
        if len(others) == 3:
            if others[0] != others[1] and others[0] != others[2] and others[1] != others[2]:
                if new_color not in others:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    all_cycles, edge_to_cycles = _precompute_edge_cycles()
    
    best_count = 0
    best_colors = [0] * NUM_EDGES
    
    # Strategy: Multiple random restarts of a greedy "maximize new colors" algorithm
    for restart in range(200):
        if time.time() > deadline:
            break
            
        random.seed(2000 + restart)
        
        # Initialize with all 0s
        colors = [0] * NUM_EDGES
        current_count = 1
        
        edges = list(range(NUM_EDGES))
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            random.shuffle(edges)
            for e in edges:
                if time.time() > deadline - 2.0:
                    break
                
                # Only try to change edges that are currently color 0
                if colors[e] != 0:
                    continue
                
                new_color = current_count
                if new_color >= NUM_COLORS:
                    break
                
                if _is_valid_change(colors, e, new_color, all_cycles, edge_to_cycles):
                    colors[e] = new_color
                    current_count += 1
                    improved = True
        
        if current_count > best_count:
            best_count = current_count
            best_colors = colors[:]
            
            # If we are doing very well, we can try to squeeze a bit more
            # by trying to change color 0 edges to new colors even if the 
            # main loop thinks it's done (maybe order dependent).
            # But the while loop already does this until no progress.

    return best_colors
