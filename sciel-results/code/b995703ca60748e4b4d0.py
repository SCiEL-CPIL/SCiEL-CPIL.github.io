"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

# Precompute adjacency masks for fast degree computation
_n = P['n']
_adj_masks = [0] * _n
for _u, _v in P['edges']:
    _adj_masks[_u] |= (1 << _v)
    _adj_masks[_v] |= (1 << _u)

def _induced_degrees(mask):
    """Compute the multiset of degrees within the induced subgraph on vertices in mask."""
    degs = []
    for v in range(_n):
        if (mask >> v) & 1:
            # Count neighbors of v that are also in mask
            d = bin(_adj_masks[v] & mask).count('1')
            degs.append(d)
    return degs

def _distinct_degree_count(mask):
    """Number of distinct degree values in the induced subgraph."""
    if mask == 0:
        return 0
    degs = _induced_degrees(mask)
    return len(set(degs))

def _max_possible_distinct(mask, current_best):
    """
    Upper bound on distinct degrees achievable by any superset of 'mask'
    (i.e., we can only add more vertices).
    The number of vertices in any superset is at most _n.
    Degrees in the induced subgraph on k vertices range from 0 to k-1.
    So the max number of distinct degrees is at most min(k, max_deg+1) where max_deg <= _n-1.
    But more tightly: if current subset already has 'cur_distinct' distinct degrees,
    and we add m more vertices, the new degrees are in [0, _n-1].
    The absolute upper bound on distinct degrees for ANY subset of size up to _n is _n.
    But we can do better: the current subset has size s. Any superset has size s' >= s.
    Degrees in a superset of size s' are in [0, s'-1]. So max distinct <= s'.
    Also, the current distinct set already occupies some values. Adding vertices can
    introduce at most (s' - s) new degree values at most (each new vertex has one degree).
    So upper bound = len(current_distinct_set) + (s' - s) where s' <= _n.
    Thus upper bound = len(current_distinct_set) + (_n - s).
    """
    s = bin(mask).count('1')
    cur_distinct = len(set(_induced_degrees(mask))) if s > 0 else 0
    # Max distinct degrees if we add all remaining vertices
    return cur_distinct + (_n - s)

def construct():
    """
    Find a subset of at least 8 vertices that maximizes the number of distinct
    vertex degrees in the induced subgraph.
    Uses bitmask branch-and-bound search.
    """
    n = _n
    full_mask = (1 << n) - 1
    
    best_count = 0
    best_mask = 0
    best_vertices = []
    
    # Precompute all subset sizes for quick reference
    # We'll do a DFS over subsets using bitmask recursion
    
    # To make the search tractable, we'll use a recursive approach that
    # considers adding or not adding each vertex, with pruning.
    
    # Order vertices by some heuristic to help pruning
    # We'll process vertices in order 0..n-1
    
    def dfs(idx, mask, size):
        nonlocal best_count, best_mask, best_vertices
        
        # Prune: if even adding all remaining vertices can't beat best
        if idx == n:
            if size >= 8:
                dc = _distinct_degree_count(mask)
                if dc > best_count:
                    best_count = dc
                    best_mask = mask
                    best_vertices = [v for v in range(n) if (mask >> v) & 1]
            return
        
        # Upper bound: current distinct + remaining vertices that can be added
        # Current distinct degrees in this partial mask
        if size > 0:
            cur_distinct = len(set(_induced_degrees(mask)))
        else:
            cur_distinct = 0
        
        # Max possible distinct if we add all remaining (n - idx) vertices
        # Each new vertex can contribute at most 1 new distinct degree value
        # But also the existing degrees might change when we add vertices!
        # This is tricky. Let's use a looser bound.
        # The total number of vertices in the final set will be between size and size + (n - idx).
        # Degrees range from 0 to (final_size - 1).
        # So max distinct <= final_size <= size + (n - idx).
        # Also, current distinct is a lower bound on final distinct only if degrees don't change,
        # but they CAN change when we add vertices.
        # So a safe upper bound is: max(size + (n - idx), cur_distinct) + (n - idx)
        # Actually, the simplest safe upper bound: the final set has at most size + (n-idx) vertices,
        # so degrees are in [0, size+(n-idx)-1], giving at most size+(n-idx) distinct values.
        upper = size + (n - idx)
        
        if upper <= best_count:
            return
        
        # Option 1: don't include vertex idx
        dfs(idx + 1, mask, size)
        
        # Option 2: include vertex idx
        new_mask = mask | (1 << idx)
        dfs(idx + 1, new_mask, size + 1)
    
    # Start DFS
    dfs(0, 0, 0)
    
    if best_count == 0:
        # Fallback: return first 8 vertices
        return list(range(8))
    
    return list(best_vertices)

# EVOLVE-BLOCK-END
