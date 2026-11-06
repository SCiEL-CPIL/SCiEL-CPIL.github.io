"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

N = 16
M = N * (N - 1) // 2  # 120 edges

# Precompute edge index
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    # i < j, index = sum_{k=0}^{i} (N-1-k) + (j-1-i)
    # = i*N - i*(i+1)//2 - 1 + (j-1-i)
    # Let's use a simpler formula
    return i * (N - 1 - i) // 2 + (j - i - 1) if i < j else 0

# Build edge list
edges = []
edge_to_uv = {}
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        edges.append((i, j))
        edge_to_uv[idx] = (i, j)
        idx += 1

# Precompute all 4-cycles: for each 4-set {a,b,c,d}, the 3 cycles
# Cycle 1: a-b-c-d-a
# Cycle 2: a-b-d-c-a
# Cycle 3: a-c-b-d-a
four_sets = list(combinations(range(N), 4))
cycles = []
for a, b, c, d in four_sets:
    # Three 4-cycles on these 4 vertices
    # Cycle 1: a-b, b-c, c-d, d-a
    # Cycle 2: a-b, b-d, d-c, c-a
    # Cycle 3: a-c, c-b, b-d, d-a
    cycles.append([(a, b), (b, c), (c, d), (d, a)])
    cycles.append([(a, b), (b, d), (d, c), (c, a)])
    cycles.append([(a, c), (c, b), (b, d), (d, a)])

# Precompute edge indices for each cycle
cycle_edge_indices = []
for cyc in cycles:
    eidx = [edge_idx(u, v) for u, v in cyc]
    cycle_edge_indices.append(eidx)

def is_valid(colors):
    """Check if no 4-cycle has 4 distinct colors."""
    for eidx in cycle_edge_indices:
        c0, c1, c2, c3 = colors[eidx[0]], colors[eidx[1]], colors[eidx[2]], colors[eidx[3]]
        if c0 != c1 and c1 != c2 and c2 != c3 and c3 != c0 and c0 != c2 and c1 != c3:
            # All four distinct
            return False
    return True

def count_colors(colors):
    return len(set(colors))

def construct():
    deadline = time.time() + 55  # 55s safety margin
    
    def check_validity(colors):
        """Fast validity check."""
        for eidx in cycle_edge_indices:
            c0, c1, c2, c3 = colors[eidx[0]], colors[eidx[1]], colors[eidx[2]], colors[eidx[3]]
            if c0 != c1 and c1 != c2 and c2 != c3 and c3 != c0 and c0 != c2 and c1 != c3:
                return False
        return True
    
    def try_add_color(colors, target_color):
        """Try to assign target_color to as many edges as possible while maintaining validity."""
        # Find edges that can take target_color
        # An edge can take target_color if it doesn't create a 4-distinct-color cycle
        # For each edge, check all cycles containing it
        # Precompute: which cycles contain which edge
        edge_to_cycles = [[] for _ in range(M)]
        for ci, eidx in enumerate(cycle_edge_indices):
            for ei in eidx:
                edge_to_cycles[ei].append(ci)
        
        # Try to add color to random edges
        new_colors = colors[:]
        used_colors = set(colors)
        if target_color in used_colors:
            return None
        
        # Try each edge in random order
        edge_order = list(range(M))
        random.shuffle(edge_order)
        
        added = 0
        for ei in edge_order:
            if new_colors[ei] == target_color:
                continue
            # Check if assigning target_color to edge ei violates any cycle
            valid = True
            for ci in edge_to_cycles[ei]:
                eidx = cycle_edge_indices[ci]
                # Check the 4 edges in this cycle
                new_c = [new_colors[x] for x in eidx]
                new_c[eidx.index(ei)] = target_color
                c0, c1, c2, c3 = new_c
                if c0 != c1 and c1 != c2 and c2 != c3 and c3 != c0 and c0 != c2 and c1 != c3:
                    valid = False
                    break
            if valid:
                new_colors[ei] = target_color
                added += 1
        
        if added > 0:
            return new_colors
        return None
    
    # Start with a simple coloring: color by (i+j) mod some number
    # This ensures many 4-cycles have repeated colors
    colors = [0] * M
    for idx, (i, j) in enumerate(edges):
        colors[idx] = (i + j) % 4
    
    best_colors = colors[:]
    best_count = count_colors(best_colors)
    
    # Try to improve by adding new colors
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Try to add a new color
        used = set(colors)
        available = [c for c in range(120) if c not in used]
        if not available:
            break
        rng.shuffle(available)
        
        improved = False
        for new_c in available:
            new_colors = try_add_color(colors, new_c)
            if new_colors is not None:
                colors = new_colors
                improved = True
                break
        
        if not improved:
            # Try local search: change random edge to a new color
            ei = rng.randint(0, M - 1)
            new_c = rng.randint(0, 119)
            if new_c != colors[ei]:
                old_c = colors[ei]
                colors[ei] = new_c
                if check_validity(colors):
                    pass  # Keep it
                else:
                    colors[ei] = old_c
        
        # Update best
        cur_count = count_colors(colors)
        if cur_count > best_count:
            best_count = cur_count
            best_colors = colors[:]
    
    return best_colors
