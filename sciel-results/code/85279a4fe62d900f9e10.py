import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
DEADLINE = 85.0

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def count_used(colors):
    return len(set(colors))

def build_edge_list():
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    return edges

def build_c4_cycles():
    """Precompute all 4-cycles (as tuples of 4 edge indices) and their 3 cycle orientations."""
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Three distinct 4-cycles on vertices {a,b,c,d}
                    # Cycle 1: a-b-c-d-a
                    e1 = (edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a))
                    # Cycle 2: a-b-d-c-a
                    e2 = (edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a))
                    # Cycle 3: a-c-b-d-a
                    e3 = (edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a))
                    cycles.append(e1)
                    cycles.append(e2)
                    cycles.append(e3)
    return cycles

def is_valid_coloring(colors):
    """Vectorized check: no 4-cycle has 4 distinct edge colors."""
    arr = np.array(colors, dtype=np.int32)
    for cyc in _CYCLES:
        c1, c2, c3, c4 = arr[cyc[0]], arr[cyc[1]], arr[cyc[2]], arr[cyc[3]]
        if c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4:
            return False
    return True

def fast_valid_after_recolor(colors, ei, ej, new_color):
    """Check validity after recoloring edge (ei,ej) to new_color.
    Only checks 4-cycles that contain edge (ei,ej)."""
    arr = np.array(colors, dtype=np.int32)
    other = [v for v in range(N) if v != ei and v != ej]
    for xi in range(len(other)):
        x = other[xi]
        for xj in range(xi + 1, len(other)):
            y = other[xj]
            # Cycle 1: ei-ej-x-y-ei
            c1 = (new_color, arr[edge_idx(ej, x)], arr[edge_idx(x, y)], arr[edge_idx(y, ei)])
            if len(set(c1)) == 4:
                return False
            # Cycle 2: ei-ej-y-x-ei
            c2 = (new_color, arr[edge_idx(ej, y)], arr[edge_idx(y, x)], arr[edge_idx(x, ei)])
            if len(set(c2)) == 4:
                return False
    return True

# Precompute cycles at module level for reuse
_CYCLES = build_c4_cycles()
_EDGE_LIST = build_edge_list()

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # Conflict-graph / branch-and-bound approach:
    # Each "assignment" is (edge_idx, color). Two assignments conflict if they are
    # on the same edge or if together with existing fixed colors they form a rainbow C4.
    # We greedily select non-conflicting assignments maximizing color diversity.
    
    # Strategy: 
    # 1. Start with a structural seed (group-based)
    # 2. Use branch-and-bound on the conflict graph to add new colors
    # 3. Each step: pick an edge, try assigning a new color, check conflicts
    
    # Phase 1: Structural initialization
    group_of = [i // 4 for i in range(N)]
    colors = [0] * NUM_EDGES
    
    for i in range(N):
        for j in range(i + 1, N):
            gi, gj = group_of[i], group_of[j]
            if gi != gj:
                colors[edge_idx(i, j)] = (gi + gj) % 2
            else:
                colors[edge_idx(i, j)] = 2
    
    if not is_valid_coloring(colors):
        colors = [0] * NUM_EDGES
    
    best_colors = list(colors)
    best_count = count_used(best_colors)
    
    # Phase 2: Branch-and-bound style expansion
    # For each edge, determine which new colors are "conflict-free"
    # Greedily add colors that maximize diversity
    
    within_group_edges = [(i, j) for i, j in _EDGE_LIST if group_of[i] == group_of[j]]
    
    def get_conflict_free_colors(colors, ei, ej, exclude_color=None):
        """Return set of colors that can be assigned to edge (ei,ej) without creating rainbow C4."""
        valid = set(range(120))
        if exclude_color is not None:
            valid.discard(exclude_color)
        
        other = [v for v in range(N) if v != ei and v != ej]
        arr = np.array(colors, dtype=np.int32)
        
        for xi in range(len(other)):
            x = other[xi]
            for xj in range(xi + 1, len(other)):
                y = other[xj]
                # For cycle ei-ej-x-y-ei: need colors[edge(ei,ej)], colors[edge(ej,x)],
                # colors[edge(x,y)], colors[edge(y,ei)] to NOT all be distinct
                c_ex = arr[edge_idx(ej, x)]
                c_xy = arr[edge_idx(x, y)]
                c_ey = arr[edge_idx(y, ei)]
                
                # new_color must equal one of c_ex, c_xy, c_ey to avoid rainbow
                allowed = {c_ex, c_xy, c_ey}
                valid &= allowed
                
                # For cycle ei-ej-y-x-ei
                c_ey2 = arr[edge_idx(ej, y)]
                c_yx = arr[edge_idx(y, x)]
                c_xe = arr[edge_idx(x, ei)]
                
                allowed2 = {c_ey2, c_yx, c_xe}
                valid &= allowed2
        
        return valid
    
    # Iterative improvement: repeatedly try to add new colors
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < DEADLINE - 5 and no_improve < max_no_improve:
        improved = False
        used = set(best_colors)
        
        # Try each within-group edge, then cross-group edges
        edges_to_try = within_group_edges + [(i,j) for i,j in _EDGE_LIST if group_of[i] != group_of[j]]
        rng.shuffle(edges_to_try)
        
        for ei, ej in edges_to_try:
            if time.time() > DEADLINE - 5:
                break
            idx = edge_idx(ei, ej)
            old_color = best_colors[idx]
            
            # Find conflict-free new colors
            cf_colors = get_conflict_free_colors(best_colors, ei, ej, exclude_color=old_color)
            
            # Prefer colors not yet used
            new_available = [c for c in cf_colors if c not in used]
            if new_available:
                # Pick a random new color
                new_color = rng.choice(new_available)
                best_colors[idx] = new_color
                used.add(new_color)
                improved = True
            elif cf_colors:
                # Only old color or already-used colors are valid; try to maintain
                pass
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
        
        # If stuck, perturb
        if no_improve >= 10:
            # Randomly change a few edges to break symmetry
            for _ in range(5):
                ei, ej = rng.choice(within_group_edges)
                c = rng.randint(0, 119)
                if fast_valid_after_recolor(best_colors, ei, ej, c):
                    best_colors[edge_idx(ei, ej)] = c
            no_improve = 0
    
    # Phase 3: Final local search pass
    while time.time() < DEADLINE - 3:
        ei, ej = rng.choice(_EDGE_LIST)
        idx = edge_idx(ei, ej)
        old_color = best_colors[idx]
        used = set(best_colors)
        
        cf_colors = get_conflict_free_colors(best_colors, ei, ej, exclude_color=old_color)
        new_available = [c for c in cf_colors if c not in used]
        
        if new_available:
            new_color = rng.choice(new_available)
            best_colors[idx] = new_color
    
    # Final verification
    if not is_valid_coloring(best_colors):
        best_colors = [0] * NUM_EDGES
    
    return best_colors
