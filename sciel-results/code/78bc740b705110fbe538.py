import time
import random
import itertools

N = 16
DEADLINE = 145.0  # safety margin before 150s kill

# Precompute all 4-subsets as bitmasks
_SUBSETS4_MASKS = []
# For each 4-subset mask, store the list of vertex indices
_SUBSETS4_VECS = []
# Precompute disjointness: for each pair of 4-masks, whether they're disjoint
_DISJOINT = None

# Precompute: for each vertex v, the list of indices i in _SUBSETS4_MASKS that contain v
_VERT_IN_SUBSETS = None

# The 24 perfect matchings of a 4x4 grid, each as tuple of (row, col) pairs
_PERM_MATCHINGS = [tuple(zip(range(4), p)) for p in itertools.permutations(range(4))]

def _init():
    global _SUBSETS4_MASKS, _SUBSETS4_VECS, _DISJOINT, _VERT_IN_SUBSETS
    if _SUBSETS4_MASKS:
        return
    for comb in itertools.combinations(range(N), 4):
        mask = 0
        for v in comb:
            mask |= (1 << v)
        _SUBSETS4_MASKS.append(mask)
        _SUBSETS4_VECS.append(comb)
    
    n4 = len(_SUBSETS4_MASKS)
    _DISJOINT = [[False] * n4 for _ in range(n4)]
    for i in range(n4):
        for j in range(i+1, n4):
            if _SUBSETS4_MASKS[i] & _SUBSETS4_MASKS[j] == 0:
                _DISJOINT[i][j] = True
                _DISJOINT[j][i] = True
    
    # For each vertex, which 4-subsets contain it
    _VERT_IN_SUBSETS = [[] for _ in range(N)]
    for i, mask in enumerate(_SUBSETS4_MASKS):
        for v in range(N):
            if (mask >> v) & 1:
                _VERT_IN_SUBSETS[v].append(i)

def _has_q3_in_bipartite(adj_masks, rows_mask, cols_mask):
    """Check if the bipartite subgraph induced by rows_mask and cols_mask contains Q3.
    Q3 = K4,4 minus a perfect matching.
    The induced graph contains Q3 iff the missing-edge graph has a perfect matching.
    """
    rows = [v for v in range(N) if (rows_mask >> v) & 1]
    cols = [v for v in range(N) if (cols_mask >> v) & 1]
    
    # For each row, compute which cols are missing
    missing_masks = []
    for r in rows:
        mm = 0
        for j, c in enumerate(cols):
            if not (adj_masks[r] >> c) & 1:
                mm |= (1 << j)
        missing_masks.append(mm)
    
    # Check if missing-edge bipartite graph has a perfect matching
    for perm in _PERM_MATCHINGS:
        ok = True
        for r_idx, c_idx in perm:
            if not (missing_masks[r_idx] >> c_idx) & 1:
                ok = False
                break
        if ok:
            return True
    return False

def _creates_q3_containing_edge(u, v, adj_masks):
    """Check if adding edge (u,v) creates a Q3 anywhere in the graph.
    We need to check all 4-subsets containing u and all 4-subsets containing v
    that are disjoint from each other.
    """
    # Get all 4-subsets containing u
    u_subsets = _VERT_IN_SUBSETS[u]
    v_subsets = _VERT_IN_SUBSETS[v]
    
    for i in u_subsets:
        rows_mask = _SUBSETS4_MASKS[i]
        for j in v_subsets:
            cols_mask = _SUBSETS4_MASKS[j]
            # They must be disjoint
            if rows_mask & cols_mask:
                continue
            # Check if this 4x4 bipartite subgraph contains Q3
            if _has_q3_in_bipartite(adj_masks, rows_mask, cols_mask):
                return True
    return False

def _is_q3_free(adj_masks):
    """Full check: does the graph contain Q3 anywhere?"""
    for i, rows_mask in enumerate(_SUBSETS4_MASKS):
        for j, cols_mask in enumerate(_SUBSETS4_MASKS):
            if not _DISJOINT[i][j]:
                continue
            if _has_q3_in_bipartite(adj_masks, rows_mask, cols_mask):
                return False
    return True

def _backtrack_build(order, seed, deadline):
    """Backtracking: assign edges for vertices in order, trying to maximize edges.
    For each vertex, decide which neighbors to connect to.
    Prune if adding any edge would create Q3.
    """
    rng = random.Random(seed)
    
    # Shuffle the order to get different results
    order = list(order)
    rng.shuffle(order)
    
    adj_masks = [0] * N
    
    # Process vertices in order
    for idx, vert in enumerate(order):
        if time.time() > deadline:
            return None
        
        # Determine which vertices are already processed
        processed_mask = 0
        for p in order[:idx]:
            processed_mask |= (1 << p)
        
        # Try to connect vert to all other vertices (in random order)
        candidates = [v for v in range(N) if v != vert]
        rng.shuffle(candidates)
        
        for other in candidates:
            if time.time() > deadline:
                return None
            
            # Tentatively add edge
            adj_masks[vert] |= (1 << other)
            adj_masks[other] |= (1 << vert)
            
            # Check if this creates Q3
            if _creates_q3_containing_edge(vert, other, adj_masks):
                # Remove edge
                adj_masks[vert] &= ~(1 << other)
                adj_masks[other] &= ~(1 << vert)
            # else: keep the edge
    
    # Final verification
    if not _is_q3_free(adj_masks):
        return None
    
    # Build edge list
    result = []
    for u in range(N):
        for v in range(u+1, N):
            if (adj_masks[u] >> v) & 1:
                result.append([u, v])
    return result

def _greedy_build(seed, deadline):
    """Greedy edge addition with random order."""
    rng = random.Random(seed)
    
    all_edges = [(u, v) for u in range(N) for v in range(u+1, N)]
    rng.shuffle(all_edges)
    
    adj_masks = [0] * N
    
    for (u, v) in all_edges:
        if time.time() > deadline:
            break
        
        # Tentatively add edge
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
        if _creates_q3_containing_edge(u, v, adj_masks):
            # Remove edge
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)
    
    # Verify
    if not _is_q3_free(adj_masks):
        return None
    
    result = []
    for u in range(N):
        for v in range(u+1, N):
            if (adj_masks[u] >> v) & 1:
                result.append([u, v])
    return result

def construct():
    _init()
    
    # Start with a known valid graph: K3,13 (39 edges)
    best_edges = [[a, b] for a in range(3) for b in range(3, 16)]
    best_count = 39
    
    seed = 0
    start_time = time.time()
    
    while time.time() < DEADLINE:
        # Alternate between backtracking and greedy
        if seed % 2 == 0:
            order = list(range(N))
            result = _backtrack_build(order, seed, DEADLINE)
        else:
            result = _greedy_build(seed, DEADLINE)
        
        if result is not None:
            cnt = len(result)
            if cnt > best_count:
                best_count = cnt
                best_edges = result
        
        seed += 1
    
    return best_edges
