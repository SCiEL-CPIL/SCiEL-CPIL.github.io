import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_idx(i, j):
    """Map (i,j) with i<j to index 0..119 in lex order."""
    # Lex order: (0,1),(0,2),...,(0,15),(1,2),(1,3),...,(14,15)
    # For vertex i, edges are (i,i+1),...,(i,N-1)
    # Index of (i,j) = sum_{k=0}^{i-1} (N-1-k) + (j-i-1)
    # = i*(N-1) - i*(i-1)//2 + (j-i-1)
    # Let me just compute it directly
    idx = 0
    for k in range(i):
        idx += N - 1 - k
    idx += j - i - 1
    return idx

# Precompute all edges
edges = []
for i in range(N):
    for j in range(i+1, N):
        edges.append((i, j))

# Precompute all 4-cycles
# A 4-cycle is defined by 4 vertices and a cyclic ordering
# For 4 vertices {a,b,c,d}, the 3 cycles are:
# (a,b,c,d), (a,b,d,c), (a,c,b,d) [up to rotation/reflection]
# Each cycle is a set of 4 edges

def get_cycles_for_4set(a, b, c, d):
    """Return the 3 cycles (as tuples of 4 edge indices) for 4 vertices a,b,c,d."""
    # Vertices in order a,b,c,d
    # Cycle 1: a-b-c-d-a: edges (a,b),(b,c),(c,d),(d,a)
    # Cycle 2: a-b-d-c-a: edges (a,b),(b,d),(d,c),(c,a)
    # Cycle 3: a-c-b-d-a: edges (a,c),(c,b),(b,d),(d,a)
    e_ab = edge_idx(min(a,b), max(a,b))
    e_bc = edge_idx(min(b,c), max(b,c))
    e_cd = edge_idx(min(c,d), max(c,d))
    e_da = edge_idx(min(d,a), max(d,a))
    e_bd = edge_idx(min(b,d), max(b,d))
    e_ca = edge_idx(min(c,a), max(c,a))
    
    cycle1 = (e_ab, e_bc, e_cd, e_da)
    cycle2 = (e_ab, e_bd, e_cd, e_ca)  # (a,b),(b,d),(d,c),(c,a) -> e_ab, e_bd, e_cd, e_ca
    cycle3 = (e_ca, e_bc, e_bd, e_da)  # (a,c),(c,b),(b,d),(d,a) -> e_ca, e_bc, e_bd, e_da
    
    return [cycle1, cycle2, cycle3]

# Build list of all cycles
all_cycles = []
cycles_by_edge = [[] for _ in range(NUM_EDGES)]  # for each edge, list of cycle indices

from itertools import combinations

for quad in combinations(range(N), 4):
    a, b, c, d = quad
    cycles = get_cycles_for_4set(a, b, c, d)
    for cyc in cycles:
        cyc_idx = len(all_cycles)
        all_cycles.append(cyc)
        for e in cyc:
            cycles_by_edge[e].append(cyc_idx)

NUM_CYCLES = len(all_cycles)
print(f"Number of cycles: {NUM_CYCLES}")

def construct():
    deadline = time.time() + 85  # 85s safety margin
    
    # Start with all zeros
    colors = [0] * NUM_EDGES
    
    # Current set of colors used
    used_colors = set(colors)
    
    def is_rainbow(cycle_idx):
        cyc = all_cycles[cycle_idx]
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        return len({c0, c1, c2, c3}) == 4
    
    def check_validity_after_change(edge_idx_changed, old_color, new_color):
        """Check if changing edge to new_color creates any rainbow cycle."""
        colors[edge_idx_changed] = new_color
        # Check all cycles containing this edge
        for cyc_idx in cycles_by_edge[edge_idx_changed]:
            if is_rainbow(cyc_idx):
                colors[edge_idx_changed] = old_color
                return False
        return True
    
    # Try to introduce new colors
    # Strategy: for each edge, try to give it a new unique color
    # If valid, keep it. Otherwise, try other edges.
    
    new_color_counter = 1  # next color to try (0 is already used)
    
    # We'll do multiple passes
    for pass_num in range(50):
        if time.time() > deadline:
            break
        
        improved = False
        for e in range(NUM_EDGES):
            if time.time() > deadline:
                break
            
            if colors[e] == 0 and new_color_counter < 120:
                # Try to give this edge a new color
                if check_validity_after_change(e, 0, new_color_counter):
                    new_color_counter += 1
                    improved = True
            elif colors[e] != 0:
                # Edge already has a color, skip
                pass
        
        if not improved:
            # Try swapping colors between edges to free up capacity
            # Or try reassigning colors
            # For now, break
            break
    
    # Count distinct colors
    distinct = len(set(colors))
    print(f"Distinct colors: {distinct}")
    
    return colors
