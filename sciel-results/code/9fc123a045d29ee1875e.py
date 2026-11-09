import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119
DEADLINE = time.time() + 80  # 80s safety margin before 90s kill

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute all 4-cycles as tuples of 4 edge indices
_CYCLES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                _CYCLES.append((_edge_idx(a,b), _edge_idx(b,c), _edge_idx(c,d), _edge_idx(d,a)))
                _CYCLES.append((_edge_idx(a,b), _edge_idx(b,d), _edge_idx(d,c), _edge_idx(c,a)))
                _CYCLES.append((_edge_idx(a,c), _edge_idx(c,b), _edge_idx(b,d), _edge_idx(d,a)))

NUM_CYCLES = len(_CYCLES)

# For each edge, precompute which cycles contain it
_EDGE_TO_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(_CYCLES):
    for e in cycle:
        _EDGE_TO_CYCLES[e].append(ci)

def _check_edge_change(colors, edge_idx, new_color):
    """Check if changing colors[edge_idx] to new_color maintains validity."""
    old_color = colors[edge_idx]
    if old_color == new_color:
        return True
    for ci in _EDGE_TO_CYCLES[edge_idx]:
        e0, e1, e2, e3 = _CYCLES[ci]
        c0 = colors[e0] if e0 != edge_idx else new_color
        c1 = colors[e1] if e1 != edge_idx else new_color
        c2 = colors[e2] if e2 != edge_idx else new_color
        c3 = colors[e3] if e3 != edge_idx else new_color
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _is_valid(colors):
    """Check if no 4-cycle has four distinct edge colors."""
    for cycle in _CYCLES:
        e0, e1, e2, e3 = cycle
        c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], colors[e3]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def _build_hierarchical(vertices, start_color, splits, color_map):
    """
    Build a hierarchical Gallai coloring.
    Returns the next available color index.
    """
    n = len(vertices)
    if n <= 1:
        return start_color
    if n == 2:
        u, v = vertices
        color_map[(min(u,v), max(u,v))] = start_color
        return start_color + 1
    if n == 3:
        u, v, w = vertices
        color_map[(min(u,v), max(u,v))] = start_color
        color_map[(min(v,w), max(v,w))] = start_color + 1
        color_map[(min(u,w), max(u,w))] = start_color + 2
        return start_color + 3
    
    k = splits.get(n, n // 2)
    v1 = vertices[:k]
    v2 = vertices[k:]
    
    cross_color = start_color
    for u in v1:
        for v in v2:
            color_map[(min(u,v), max(u,v))] = cross_color
    
    next_color = start_color + 1
    next_color = _build_hierarchical(v1, next_color, splits, color_map)
    next_color = _build_hierarchical(v2, next_color, splits, color_map)
    return next_color

def _generate_hierarchical(splits):
    """Generate a hierarchical coloring and return it as a flat list."""
    color_map = {}
    vertices = list(range(N))
    _build_hierarchical(vertices, 0, splits, color_map)
    
    result = []
    for i in range(N):
        for j in range(i+1, N):
            result.append(color_map[(i,j)])
    return result

def _local_search(colors, max_iterations=500000):
    """
    Local search: try to increase the number of distinct colors by changing
    individual edge colors to unused colors.
    """
    colors = list(colors)
    current_count = _count_colors(colors)
    used_colors = set(colors)
    unused_colors = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
    
    if not unused_colors:
        return colors, current_count
    
    random.shuffle(unused_colors)
    
    # Try multiple passes
    for iteration in range(max_iterations):
        if time.time() > DEADLINE:
            break
        
        improved = False
        # Shuffle edges each pass for diversity
        edge_order = list(range(NUM_EDGES))
        random.shuffle(edge_order)
        
        for eidx in edge_order:
            if time.time() > DEADLINE:
                break
            
            old_color = colors[eidx]
            # Try changing to an unused color
            for new_color in unused_colors[:20]:  # Try first 20 unused colors
                if _check_edge_change(colors, eidx, new_color):
                    colors[eidx] = new_color
                    used_colors.add(new_color)
                    if new_color in unused_colors:
                        unused_colors.remove(new_color)
                    current_count = _count_colors(colors)
                    improved = True
                    break
        
        if not improved:
            break
    
    return colors, current_count

def construct():
    """
    Returns a list of 120 integer colors for the edges of K_16.
    Uses hierarchical Gallai coloring as a starting point, then applies
    local search to maximize the number of distinct colors used.
    """
    # Optimal splits from inspiration 1
    SPLIT = {
        4: 3, 5: 3, 6: 3, 7: 3, 8: 4, 9: 3, 10: 4,
        11: 5, 12: 6, 13: 6, 14: 7, 15: 6, 16: 7
    }
    
    # Generate the base hierarchical coloring
    base_colors = _generate_hierarchical(SPLIT)
    base_count = _count_colors(base_colors)
    
    # Try local search from the base coloring
    best_colors = list(base_colors)
    best_count = base_count
    
    # Multiple random restarts of local search with different seeds
    for seed in range(10):
        if time.time() > DEADLINE:
            break
        
        random.seed(seed)
        search_colors, search_count = _local_search(list(base_colors), max_iterations=200000)
        
        if search_count > best_count:
            best_count = search_count
            best_colors = search_colors
    
    # Final validation
    if not _is_valid(best_colors):
        # Fallback to base coloring if something went wrong
        best_colors = list(base_colors)
        best_count = base_count
    
    return best_colors
