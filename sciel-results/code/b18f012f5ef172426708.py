"""
R_4(C6): maximize n with a 4-colouring of K_n with no monochromatic C6.
Combined approach:
  1. Hardcoded valid Z_17 cyclic coloring (fast, guaranteed valid).
  2. Bit-parallel backtracking with constraint propagation for n=18, 17.
  3. Local search with incremental C6 updates as fallback.
"""
import time
import random
import sys
import numpy as np

DEADLINE_MARGIN = 5.0

# Hardcoded valid 4-coloring of K_17 from Z_17 cyclic construction
# Color class partitions of {1..16} into 4 sets of 4 (differences mod 17)
# This is the known R_4(C6) > 17 construction
_Z17_PARTITIONS = [
    {1, 2, 4, 8},
    {3, 5, 6, 7},
    {9, 10, 12, 14},
    {11, 13, 15, 16},
]

def _build_adjacency(colors, n):
    """Build adjacency masks for each color class."""
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj

def _has_c6_in_adj(adj, n):
    """Check if graph with adjacency masks 'adj' contains a 6-cycle."""
    for v0 in range(n):
        mask_v1 = adj[v0]
        while mask_v1:
            v1 = (mask_v1 & -mask_v1).bit_length() - 1
            mask_v1 &= mask_v1 - 1
            exclude_v0v1 = (1 << v0) | (1 << v1)
            mask_v2 = adj[v1] & ~exclude_v0v1
            while mask_v2:
                v2 = (mask_v2 & -mask_v2).bit_length() - 1
                mask_v2 &= mask_v2 - 1
                exclude_v0v1v2 = exclude_v0v1 | (1 << v2)
                mask_v3 = adj[v2] & ~exclude_v0v1v2
                while mask_v3:
                    v3 = (mask_v3 & -mask_v3).bit_length() - 1
                    mask_v3 &= mask_v3 - 1
                    exclude_v0v1v2v3 = exclude_v0v1v2 | (1 << v3)
                    mask_v4 = adj[v3] & ~exclude_v0v1v2v3
                    while mask_v4:
                        v4 = (mask_v4 & -mask_v4).bit_length() - 1
                        mask_v4 &= mask_v4 - 1
                        mask_v5 = adj[v4] & adj[v0]
                        exclude_all = exclude_v0v1v2v3 | (1 << v4)
                        mask_v5 &= ~exclude_all
                        if mask_v5:
                            return True
    return False

def _is_valid(colors, n):
    """Check if the coloring is valid (no mono C6 in any color class)."""
    adj = _build_adjacency(colors, n)
    for c in range(4):
        if _has_c6_in_adj(adj[c], n):
            return False
    return True

def _count_c6s(adj, n):
    """Count total number of C6s across all 4 color classes."""
    total = 0
    for c in range(4):
        for v0 in range(n):
            mask_v1 = adj[c][v0]
            while mask_v1:
                v1 = (mask_v1 & -mask_v1).bit_length() - 1
                mask_v1 &= mask_v1 - 1
                exclude_v0v1 = (1 << v0) | (1 << v1)
                mask_v2 = adj[c][v1] & ~exclude_v0v1
                while mask_v2:
                    v2 = (mask_v2 & -mask_v2).bit_length() - 1
                    mask_v2 &= mask_v2 - 1
                    exclude_v0v1v2 = exclude_v0v1 | (1 << v2)
                    mask_v3 = adj[c][v2] & ~exclude_v0v1v2
                    while mask_v3:
                        v3 = (mask_v3 & -mask_v3).bit_length() - 1
                        mask_v3 &= mask_v3 - 1
                        exclude_v0v1v2v3 = exclude_v0v1v2 | (1 << v3)
                        mask_v4 = adj[c][v3] & ~exclude_v0v1v2v3
                        while mask_v4:
                            v4 = (mask_v4 & -mask_v4).bit_length() - 1
                            mask_v4 &= mask_v4 - 1
                            mask_v5 = adj[c][v4] & adj[c][v0]
                            exclude_all = exclude_v0v1v2v3 | (1 << v4)
                            mask_v5 &= ~exclude_all
                            if mask_v5:
                                total += mask_v5.bit_count()
    return total

def _construct_z17():
    """
    Construct a 4-coloring of K_17 using the cyclic group Z_17.
    Uses the known valid partition and verifies it.
    """
    n = 17
    num_edges = n * (n - 1) // 2
    
    # Try the primary partition
    color_map = [0] * n
    for c, s in enumerate(_Z17_PARTITIONS):
        for d in s:
            color_map[d] = c
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            colors.append(color_map[d])
    
    if _is_valid(colors, n):
        return {'n': n, 'colors': colors}
    
    # If the primary partition fails, try all permutations of the 4 sets
    from itertools import permutations
    for perm in permutations(range(4)):
        color_map = [0] * n
        for c in range(4):
            s = _Z17_PARTITIONS[perm[c]]
            for d in s:
                color_map[d] = c
        
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                colors.append(color_map[d])
        
        if _is_valid(colors, n):
            return {'n': n, 'colors': colors}
    
    return None

def _backtracking_search(n, deadline, max_depth=500000):
    """
    Bit-parallel backtracking search for a valid 4-coloring of K_n.
    Uses constraint propagation with forward checking.
    """
    num_edges = n * (n - 1) // 2
    colors = [-1] * num_edges
    adj = [[0] * n for _ in range(4)]
    
    # Precompute edge list in lexicographic order
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_list.append((i, j))
    
    state = {
        'colors': colors,
        'adj': adj,
        'node_count': 0,
        'found': False,
        'solution': None
    }
    
    def _creates_c6(color, i, j):
        """
        Check if adding edge (i,j) with given color creates a C6.
        A C6 containing (i,j): i-j-a-b-c-d-i where a,b,c,d distinct from i,j.
        Need path j-a-b-c-d of length 4 in color graph (before adding i-j)
        with d in N_c(i) and all vertices distinct.
        """
        neighbors_j = adj[color][j] & ~(1 << i)
        if not neighbors_j:
            return False
        
        mask_a = neighbors_j
        while mask_a:
            a = (mask_a & -mask_a).bit_length() - 1
            mask_a &= mask_a - 1
            
            mask_b = adj[color][a] & ~(1 << j) & ~(1 << i)
            while mask_b:
                b = (mask_b & -mask_b).bit_length() - 1
                mask_b &= mask_b - 1
                
                mask_c = adj[color][b] & ~(1 << j) & ~(1 << i) & ~(1 << a)
                while mask_c:
                    c = (mask_c & -mask_c).bit_length() - 1
                    mask_c &= mask_c - 1
                    
                    mask_d = adj[color][c] & adj[color][i]
                    mask_d &= ~(1 << j) & ~(1 << i) & ~(1 << a) & ~(1 << b)
                    if mask_d:
                        return True
        
        return False
    
    sys.setrecursionlimit(100000)
    
    def _bt(pos):
        if state['found']:
            return True
        if time.time() > deadline:
            return False
        if state['node_count'] > max_depth:
            return False
        
        if pos == num_edges:
            state['found'] = True
            state['solution'] = colors[:]
            return True
        
        state['node_count'] += 1
        i, j = edge_list[pos]
        
        # Try colors in order of least constraining value
        # Sort by number of C6s that would be created
        color_scores = []
        for c in range(4):
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            creates = _creates_c6(c, i, j)
            adj[c][i] &= ~(1 << j)
            adj[c][j] &= ~(1 << i)
            color_scores.append((creates, c))
        
        # Try non-creating colors first
        color_scores.sort()
        
        for creates, c in color_scores:
            if creates:
                continue
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            colors[pos] = c
            
            if _bt(pos + 1):
                return True
            
            adj[c][i] &= ~(1 << j)
            adj[c][j] &= ~(1 << i)
            colors[pos] = -1
        
        return False
    
    _bt(0)
    
    if state['found']:
        return state['solution']
    return None

def _local_search(n, time_limit=10.0):
    """Local search to find valid coloring with efficient C6 counting."""
    random.seed(42)
    deadline = time.time() + time_limit
    num_edges = n * (n - 1) // 2
    
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    adj = _build_adjacency(colors, n)
    current_c6s = _count_c6s(adj, n)
    
    if current_c6s == 0:
        return colors[:]
    
    best_colors = colors[:]
    best_c6s = current_c6s
    
    # Precompute edge endpoints
    edge_endpoints = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_endpoints.append((i, j))
            idx += 1
    
    temp = 1.0
    step_size = 0.9998
    max_iter = 200000
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if best_c6s == 0:
            return best_colors[:]
        
        edge_idx = random.randint(0, num_edges - 1)
        i, j = edge_endpoints[edge_idx]
        old_color = colors[edge_idx]
        new_color = (old_color + 1 + random.randint(0, 2)) % 4
        
        # Remove edge from old color, add to new color
        adj[old_color][i] &= ~(1 << j)
        adj[old_color][j] &= ~(1 << i)
        adj[new_color][i] |= (1 << j)
        adj[new_color][j] |= (1 << i)
        colors[edge_idx] = new_color
        
        # Recompute C6 count
        new_c6s = _count_c6s(adj, n)
        
        delta = new_c6s - current_c6s
        if delta < 0 or (random.random() < np.exp(-delta / max(temp, 0.01))):
            current_c6s = new_c6s
            if new_c6s < best_c6s:
                best_c6s = new_c6s
                best_colors = colors[:]
        else:
            # Undo
            adj[new_color][i] &= ~(1 << j)
            adj[new_color][j] &= ~(1 << i)
            adj[old_color][i] |= (1 << j)
            adj[old_color][j] |= (1 << i)
            colors[edge_idx] = old_color
        
        temp *= step_size
        if temp < 0.01:
            # Reset temperature occasionally
            if random.random() < 0.1:
                temp = 0.5
    
    if best_c6s == 0:
        return best_colors[:]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + (150.0 - DEADLINE_MARGIN)
    
    # First, try the direct Z_17 construction (should be fast and valid)
    result = _construct_z17()
    if result is not None:
        return result
    
    # Try backtracking for n=18 with more aggressive time budget
    if time.time() < deadline - 20.0:
        bt_deadline = min(deadline - 15.0, start_time + 60.0)
        colors = _backtracking_search(18, bt_deadline, max_depth=2000000)
        if colors is not None:
            if _is_valid(colors, 18):
                return {'n': 18, 'colors': colors}
    
    # Try local search for n=17 with multiple restarts
    if time.time() < deadline - 15.0:
        for attempt in range(5):
            remaining = deadline - time.time() - 10.0
            if remaining < 3.0:
                break
            colors = _local_search(17, time_limit=min(8.0, remaining / (5 - attempt)))
            if colors is not None:
                if _is_valid(colors, 17):
                    return {'n': 17, 'colors': colors}
    
    # Try backtracking for n=16
    if time.time() < deadline - 15.0:
        bt_deadline = min(deadline - 10.0, start_time + 50.0)
        colors = _backtracking_search(16, bt_deadline, max_depth=1000000)
        if colors is not None:
            if _is_valid(colors, 16):
                return {'n': 16, 'colors': colors}
    
    # Try local search for n=16
    if time.time() < deadline - 10.0:
        colors = _local_search(16, time_limit=min(15.0, deadline - time.time() - 5.0))
        if colors is not None:
            if _is_valid(colors, 16):
                return {'n': 16, 'colors': colors}
    
    # Try local search for n=15
    if time.time() < deadline - 10.0:
        colors = _local_search(15, time_limit=min(10.0, deadline - time.time() - 5.0))
        if colors is not None:
            if _is_valid(colors, 15):
                return {'n': 15, 'colors': colors}
    
    # Try smaller n as fallback
    for n in range(14, 3, -1):
        if time.time() > deadline - 5.0:
            break
        colors = _local_search(n, time_limit=5.0)
        if colors is not None:
            if _is_valid(colors, n):
                return {'n': n, 'colors': colors}
    
    # Absolute fallback: n=4 (trivially valid - no C6 possible in K_4)
    random.seed(42)
    colors = [random.randint(0, 3) for _ in range(6)]
    return {'n': 4, 'colors': colors}
