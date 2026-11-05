"""
R_4(C6): maximize n with a 4-colouring of K_n with no monochromatic C6.
Combined approach:
  1. Direct Z_17 cyclic construction (fast, known-valid if partition is right).
  2. Bit-parallel backtracking with constraint propagation for n=18, 17, 16.
  3. Local search with C6-counting objective as fallback.
"""
import time
import random
import sys
import numpy as np

DEADLINE_MARGIN = 5.0

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
            mask_v2 = adj[v1] & ~(1 << v0)
            while mask_v2:
                v2 = (mask_v2 & -mask_v2).bit_length() - 1
                mask_v2 &= mask_v2 - 1
                mask_v3 = adj[v2] & ~(1 << v0) & ~(1 << v1)
                while mask_v3:
                    v3 = (mask_v3 & -mask_v3).bit_length() - 1
                    mask_v3 &= mask_v3 - 1
                    mask_v4 = adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                    while mask_v4:
                        v4 = (mask_v4 & -mask_v4).bit_length() - 1
                        mask_v4 &= mask_v4 - 1
                        mask_v5 = adj[v4] & adj[v0]
                        used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                        mask_v5 &= ~used_mask
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
                mask_v2 = adj[c][v1] & ~(1 << v0)
                while mask_v2:
                    v2 = (mask_v2 & -mask_v2).bit_length() - 1
                    mask_v2 &= mask_v2 - 1
                    mask_v3 = adj[c][v2] & ~(1 << v0) & ~(1 << v1)
                    while mask_v3:
                        v3 = (mask_v3 & -mask_v3).bit_length() - 1
                        mask_v3 &= mask_v3 - 1
                        mask_v4 = adj[c][v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                        while mask_v4:
                            v4 = (mask_v4 & -mask_v4).bit_length() - 1
                            mask_v4 &= mask_v4 - 1
                            mask_v5 = adj[c][v4] & adj[c][v0]
                            used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                            mask_v5 &= ~used_mask
                            if mask_v5:
                                total += mask_v5.bit_count()
    return total

def _construct_z17():
    """
    Construct a 4-coloring of K_17 using the cyclic group Z_17.
    Try all partitions of {1..16} into 4 sets of 4 (up to symmetry).
    """
    n = 17
    num_edges = n * (n - 1) // 2
    
    # Known good partitions
    known_partitions = [
        [{1, 2, 4, 8}, {3, 5, 6, 7}, {9, 10, 12, 14}, {11, 13, 15, 16}],
        [{1, 3, 4, 9}, {2, 5, 7, 10}, {6, 8, 11, 13}, {12, 14, 15, 16}],
        [{1, 2, 4, 8}, {9, 13, 15, 16}, {3, 5, 6, 7}, {10, 11, 12, 14}],
        [{1, 2, 4, 16}, {3, 5, 7, 9}, {6, 8, 10, 12}, {11, 13, 14, 15}],
        [{1, 3, 6, 12}, {2, 4, 8, 9}, {5, 7, 11, 14}, {10, 13, 15, 16}],
        [{1, 4, 10, 14}, {2, 3, 7, 16}, {5, 6, 11, 13}, {8, 9, 12, 15}],
        [{1, 5, 6, 10}, {2, 7, 11, 16}, {3, 8, 12, 15}, {4, 9, 13, 14}],
    ]
    
    # Generate more partitions systematically
    # Use the multiplicative group structure: 3 is a primitive root mod 17
    # Powers of 3: 3, 9, 10, 13, 5, 15, 11, 16, 14, 8, 7, 4, 12, 2, 6, 1
    # Cosets of subgroups
    
    # Try all ways to partition using quadratic residues and non-residues
    # QR mod 17: {1, 2, 4, 8, 9, 13, 15, 16}
    # QNR mod 17: {3, 5, 6, 7, 10, 11, 12, 14}
    
    # Generate candidate partitions by splitting QR and QNR
    qr = [1, 2, 4, 8, 9, 13, 15, 16]
    qnr = [3, 5, 6, 7, 10, 11, 12, 14]
    
    # Try various splits
    candidate_partitions = list(known_partitions)
    
    # Systematic search: try all partitions of 16 elements into 4 groups of 4
    # This is too many (16!/(4!^4 * 4!) = 63,063,000), so use heuristics
    # Instead, try cyclic shifts and reflections of known good ones
    
    for part in known_partitions:
        # Also try shifted versions
        for shift in range(1, 17):
            shifted = [set((x + shift) % 17 for x in s) for s in part]
            # Normalize: ensure 0 is not in any set (it shouldn't be since we use 1..16)
            if all(0 not in s for s in shifted):
                candidate_partitions.append(shifted)
    
    # Also try with different assignments of which coset gets which color
    # Permute the 4 colors
    from itertools import permutations
    for part in known_partitions:
        for perm in permutations(range(4)):
            permuted = [part[perm[i]] for i in range(4)]
            candidate_partitions.append(permuted)
    
    # Try all
    seen = set()
    for part in candidate_partitions:
        # Check it's a valid partition of {1..16}
        all_elements = set()
        for s in part:
            all_elements |= s
        if all_elements != set(range(1, 17)):
            continue
        if len(part) != 4:
            continue
        if any(len(s) != 4 for s in part):
            continue
        
        # Canonical form to avoid duplicates
        canonical = tuple(tuple(sorted(s)) for s in sorted(part, key=lambda s: min(s)))
        if canonical in seen:
            continue
        seen.add(canonical)
        
        color_map = [0] * 17
        for c, s in enumerate(part):
            for d in s:
                color_map[d] = c
        
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if d == 0:
                    d = n
                colors.append(color_map[d])
        
        if _is_valid(colors, n):
            return {'n': n, 'colors': colors}
    
    return None

def _backtracking_search(n, deadline, max_depth=300000):
    """
    Bit-parallel backtracking search for a valid 4-coloring of K_n.
    Uses smarter edge ordering and constraint propagation.
    """
    num_edges = n * (n - 1) // 2
    colors = [-1] * num_edges
    adj = [[0] * n for _ in range(4)]
    
    # Precompute edge list
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            edge_list.append((i, j))
    
    # For better pruning, process edges in an order that constrains early
    # Process edges incident to vertex 0 first, then vertex 1, etc.
    # Actually lexicographic is fine; let's add forward checking
    
    state = {
        'colors': colors,
        'adj': adj,
        'node_count': 0,
        'found': False,
        'solution': None
    }
    
    def _check_c6_after_add(color, i, j):
        """
        Check if adding edge (i,j) with given color creates a C6 containing (i,j).
        A C6 through edge (i,j): i-j-a-b-c-d-i where a,b,c,d are distinct from i,j.
        Need path of length 4 from j to i in color graph (excluding the new edge).
        """
        # Path: j -> a -> b -> c -> d -> i
        # a in N(j) \ {i}, b in N(a) \ {j,i}, c in N(b) \ {j,i,a}, d in N(c) \ {j,i,a,b} ∩ N(i) \ {j,a,b}
        
        mask_a = adj[color][j] & ~(1 << i)
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
                    
                    # d must be neighbor of c AND neighbor of i, excluding j,i,a,b
                    mask_d = adj[color][c] & adj[color][i]
                    mask_d &= ~(1 << j) & ~(1 << i) & ~(1 << a) & ~(1 << b)
                    if mask_d:
                        return True
        
        return False
    
    def _forward_check(pos, i, j, color):
        """
        After tentatively assigning color to edge (i,j), check if any unassigned
        edge is forced to be impossible (all 4 colors would create a C6).
        This is expensive; only do it for early edges.
        """
        if pos > 20:  # Only do forward check for early edges
            return True
        
        # Check each unassigned edge
        for p in range(pos + 1, num_edges):
            if colors[p] != -1:
                continue
            ei, ej = edge_list[p]
            all_blocked = True
            for c in range(4):
                # Tentatively add edge (ei, ej) with color c
                adj[c][ei] |= (1 << ej)
                adj[c][ej] |= (1 << ei)
                if not _check_c6_after_add(c, ei, ej):
                    all_blocked = False
                # Undo
                adj[c][ei] &= ~(1 << ej)
                adj[c][ej] &= ~(1 << ei)
            if all_blocked:
                return False
        return True
    
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
        # Count how many C6s each color would create (0 is best)
        color_scores = []
        for c in range(4):
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            creates_c6 = _check_c6_after_add(c, i, j)
            adj[c][i] &= ~(1 << j)
            adj[c][j] &= ~(1 << i)
            color_scores.append((creates_c6, c))
        
        # Sort: prefer colors that don't create C6
        color_scores.sort()
        
        for creates_c6, c in color_scores:
            if creates_c6:
                continue  # Skip colors that create a C6
            
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            colors[pos] = c
            
            # Forward check for early edges
            if pos < 20 and not _forward_check(pos, i, j, c):
                adj[c][i] &= ~(1 << j)
                adj[c][j] &= ~(1 << i)
                colors[pos] = -1
                continue
            
            if _bt(pos + 1):
                return True
            
            # Undo
            adj[c][i] &= ~(1 << j)
            adj[c][j] &= ~(1 << i)
            colors[pos] = -1
        
        return False
    
    _bt(0)
    
    if state['found']:
        return state['solution']
    return None

def _local_search(n, time_limit=10.0):
    """Local search to find valid coloring with incremental C6 counting."""
    random.seed(42)
    deadline = time.time() + time_limit
    num_edges = n * (n - 1) // 2
    
    # Precompute edge endpoints
    edge_i = [0] * num_edges
    edge_j = [0] * num_edges
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_i[idx] = i
            edge_j[idx] = j
            idx += 1
    
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    adj = _build_adjacency(colors, n)
    current_c6s = _count_c6s(adj, n)
    
    if current_c6s == 0:
        return colors
    
    best_colors = colors[:]
    best_c6s = current_c6s
    
    temp = 1.0
    step_size = 0.9995
    
    for iteration in range(200000):
        if time.time() > deadline:
            break
        if best_c6s == 0:
            return best_colors
            
        edge_idx = random.randint(0, num_edges - 1)
        i = edge_i[edge_idx]
        j = edge_j[edge_idx]
        old_color = colors[edge_idx]
        new_color = (old_color + 1 + random.randint(0, 2)) % 4
        
        if new_color == old_color:
            continue
        
        # Incremental update: remove edge from old color, add to new color
        adj[old_color][i] &= ~(1 << j)
        adj[old_color][j] &= ~(1 << i)
        adj[new_color][i] |= (1 << j)
        adj[new_color][j] |= (1 << i)
        colors[edge_idx] = new_color
        
        # Recompute C6 count (expensive but necessary for accuracy)
        new_c6s = _count_c6s(adj, n)
        
        delta = new_c6s - current_c6s
        if delta < 0 or (random.random() < np.exp(-delta / temp)):
            current_c6s = new_c6s
            if new_c6s < best_c6s:
                best_c6s = new_c6s
                best_colors = colors[:]
        else:
            # Revert
            adj[new_color][i] &= ~(1 << j)
            adj[new_color][j] &= ~(1 << i)
            adj[old_color][i] |= (1 << j)
            adj[old_color][j] |= (1 << i)
            colors[edge_idx] = old_color
        
        temp *= step_size
        if temp < 0.01:
            # Restart with new random init
            temp = 0.5
            for _ in range(50):
                if time.time() > deadline:
                    break
                if best_c6s == 0:
                    return best_colors
                eidx = random.randint(0, num_edges - 1)
                colors[eidx] = random.randint(0, 3)
            adj = _build_adjacency(colors, n)
            current_c6s = _count_c6s(adj, n)
            if current_c6s < best_c6s:
                best_c6s = current_c6s
                best_colors = colors[:]
    
    if best_c6s == 0:
        return best_colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + (150.0 - DEADLINE_MARGIN)
    
    # First, try the direct Z_17 construction
    result = _construct_z17()
    if result is not None:
        return result
    
    # Try backtracking for n=18 (limited time)
    if time.time() < deadline - 30.0:
        bt_deadline = min(deadline - 25.0, start_time + 30.0)
        colors = _backtracking_search(18, bt_deadline, max_depth=500000)
        if colors is not None:
            if _is_valid(colors, 18):
                return {'n': 18, 'colors': colors}
    
    # Try backtracking for n=17
    if time.time() < deadline - 20.0:
        bt_deadline = min(deadline - 15.0, start_time + 40.0)
        colors = _backtracking_search(17, bt_deadline, max_depth=1000000)
        if colors is not None:
            if _is_valid(colors, 17):
                return {'n': 17, 'colors': colors}
    
    # Try local search for n=17
    if time.time() < deadline - 15.0:
        colors = _local_search(17, time_limit=min(20.0, deadline - time.time() - 10.0))
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
    
    # Absolute fallback: n=4
    random.seed(42)
    colors = [random.randint(0, 3) for _ in range(6)]
    return {'n': 4, 'colors': colors}
