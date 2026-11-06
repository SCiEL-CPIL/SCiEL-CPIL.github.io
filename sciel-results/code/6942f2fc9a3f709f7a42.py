"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
DEADLINE = 85.0  # safety margin before 90s kill

# Precompute all 4-cycles: for each 4-vertex subset, the 3 cycles
def build_cycles():
    """Return list of cycles, each cycle is a list of 4 edge indices."""
    # Map (i,j) with i<j to edge index
    edge_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_idx[(i, j)] = idx
            idx += 1
    def eidx(i, j):
        if i > j:
            i, j = j, i
        return edge_idx[(i, j)]
    
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a
                    cycles.append([eidx(a,b), eidx(b,c), eidx(c,d), eidx(d,a)])
                    # Cycle 2: a-b-d-c-a
                    cycles.append([eidx(a,b), eidx(b,d), eidx(d,c), eidx(c,a)])
                    # Cycle 3: a-c-b-d-a
                    cycles.append([eidx(a,c), eidx(c,b), eidx(b,d), eidx(d,a)])
    return cycles

def count_distinct_colors(colors):
    return len(set(colors))

def is_valid(colors):
    """Check no 4-cycle has 4 distinct colors."""
    for cyc in _CYCLES:
        if len(set(colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]])) == 4:
            return False
    return True

def check_validity_fast(colors, cycles):
    """Fast validity check using numpy."""
    # For each cycle, check if all 4 colors are distinct
    arr = np.array(colors, dtype=np.int32)
    for cyc in cycles:
        c0, c1, c2, c3 = arr[cyc[0]], arr[cyc[1]], arr[cyc[2]], arr[cyc[3]]
        if len(set([c0, c1, c2, c3])) == 4:
            return False
    return True

# Precompute cycles once
_CYCLES = build_cycles()

def initial_coloring():
    """Linear ordering: c(i,j) = j"""
    colors = [0] * NUM_EDGES
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            colors[idx] = j
            idx += 1
    return colors

def local_search(colors, max_time=80.0):
    """Local search to maximize distinct colors while maintaining validity."""
    start_time = time.time()
    colors = list(colors)
    best_colors = list(colors)
    best_count = count_distinct_colors(colors)
    
    # Build edge index mapping
    edge_idx = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_idx[(i, j)] = idx
            idx += 1
    def eidx(i, j):
        if i > j:
            i, j = j, i
        return edge_idx[(i, j)]
    
    # For each edge, precompute which cycles it belongs to
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(_CYCLES):
        for ei in cyc:
            edge_to_cycles[ei].append(ci)
    
    # Try to increase colors by changing edges
    random.seed(42)
    for iteration in range(2000000):
        if time.time() - start_time > max_time:
            break
        if time.time() > DEADLINE:
            break
        
        # Pick a random edge
        e = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Try to assign a new color
        # First try to use a color not currently in use
        used_colors = set(colors)
        available = [c for c in range(120) if c not in used_colors]
        
        if available:
            new_color = random.choice(available)
        else:
            # Try a random color
            new_color = random.randint(0, 119)
            if new_color == old_color:
                continue
        
        # Check validity: only cycles containing edge e need to be re-checked
        # But changing edge e affects all cycles containing e
        colors[e] = new_color
        valid = True
        for ci in edge_to_cycles[e]:
            cyc = _CYCLES[ci]
            if len(set(colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]])) == 4:
                valid = False
                break
        
        if valid:
            new_count = count_distinct_colors(colors)
            if new_count > best_count:
                best_count = new_count
                best_colors = list(colors)
        else:
            colors[e] = old_color
    
    return best_colors, best_count

def construct():
    start_time = time.time()
    
    # Start with linear ordering
    colors = initial_coloring()
    base_count = count_distinct_colors(colors)
    
    # Local search to improve
    best_colors, best_count = local_search(colors, max_time=80.0)
    
    if time.time() > DEADLINE:
        return best_colors
    
    # Try multiple random restarts if time permits
    for restart in range(10):
        if time.time() > DEADLINE - 5:
            break
        
        # Create a perturbed starting point
        perturbed = list(best_colors)
        for _ in range(10):
            e = random.randint(0, NUM_EDGES - 1)
            perturbed[e] = random.randint(0, 119)
        
        # Check if perturbed is valid; if not, revert changes one by one
        # Actually, just run local search from perturbed
        result, count = local_search(perturbed, max_time=75.0 - (time.time() - start_time))
        
        if count > best_count:
            best_count = count
            best_colors = result
    
    return best_colors
