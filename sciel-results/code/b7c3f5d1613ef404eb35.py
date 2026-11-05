import time
import random
from itertools import combinations

DEADLINE = 145.0
START = time.time()

# Precomputed popcount table for 26-bit integers (we only need up to 2^26)
# But we can use a smaller table and split
_POPCOUNT_TABLE = [0] * 65536
for _i in range(1, 65536):
    _POPCOUNT_TABLE[_i] = _POPCOUNT_TABLE[_i >> 1] + (_i & 1)

def _popcount_fast(x):
    """Fast popcount using table lookup."""
    return _POPCOUNT_TABLE[x & 0xFFFF] + _POPCOUNT_TABLE[(x >> 16) & 0xFFFF]

def _is_non_2colorable(n, edges):
    """
    Returns True if the hypergraph is NOT 2-colorable.
    Uses backtracking with unit propagation.
    edges: list of lists of 5 vertices.
    """
    m = len(edges)
    if m == 0:
        return False
    
    # Precompute edge masks
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # Precompute vertex to edge index mapping
    vert_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vert_edges[v].append(i)
    
    # Cache for popcount results within this call
    _pc_cache = {}
    
    def popcount(x):
        if x in _pc_cache:
            return _pc_cache[x]
        r = _popcount_fast(x)
        _pc_cache[x] = r
        return r

    def propagate(mask0, mask1):
        """
        Returns (new_mask0, new_mask1, success)
        If success is False, contradiction found.
        """
        changed = True
        while changed:
            changed = False
            for e_mask in edge_masks:
                c0 = popcount(e_mask & mask0)
                c1 = popcount(e_mask & mask1)
                c_un = 5 - c0 - c1
                
                if c0 == 5 or c1 == 5:
                    return 0, 0, False # Contradiction
                
                if c0 == 4 and c_un == 1:
                    # The unassigned vertex in this edge must be 1
                    unassigned_mask = e_mask & ~(mask0 | mask1)
                    new_mask1 = mask1 | unassigned_mask
                    if new_mask1 != mask1:
                        mask1 = new_mask1
                        changed = True
                elif c1 == 4 and c_un == 1:
                    # The unassigned vertex in this edge must be 0
                    unassigned_mask = e_mask & ~(mask0 | mask1)
                    new_mask0 = mask0 | unassigned_mask
                    if new_mask0 != mask0:
                        mask0 = new_mask0
                        changed = True
        return mask0, mask1, True

    def backtrack(mask0, mask1):
        # Find an unassigned vertex to branch on
        best_v = -1
        best_score = -1
        
        unassigned_all = ~(mask0 | mask1) & ((1 << n) - 1)
        
        while unassigned_all:
            lsb = unassigned_all & -unassigned_all
            v = lsb.bit_length() - 1
            unassigned_all ^= lsb
            
            score = 0
            for ei in vert_edges[v]:
                e_mask = edge_masks[ei]
                c0 = popcount(e_mask & mask0)
                c1 = popcount(e_mask & mask1)
                c_un = 5 - c0 - c1
                
                if c_un == 1:
                    score += 10
                elif c_un == 2:
                    score += 5
                elif c_un == 3:
                    score += 2
                elif c_un == 4:
                    score += 1
                
            if score > best_score:
                best_score = score
                best_v = v
        
        if best_v == -1:
            for e_mask in edge_masks:
                c0 = popcount(e_mask & mask0)
                c1 = popcount(e_mask & mask1)
                if c0 == 5 or c1 == 5:
                    return True
            return False
        
        # Branch on best_v
        new_mask0 = mask0 | (1 << best_v)
        if not (new_mask0 & mask1):
            nm0, nm1, ok = propagate(new_mask0, mask1)
            if ok:
                if backtrack(nm0, nm1):
                    return True
        
        new_mask1 = mask1 | (1 << best_v)
        if not (new_mask1 & mask0):
            nm0, nm1, ok = propagate(mask0, new_mask1)
            if ok:
                if backtrack(nm0, nm1):
                    return True
        
        return True

    nm0, nm1, ok = propagate(0, 0)
    if not ok:
        return True
    
    return backtrack(nm0, nm1)

def _try_prune(seed, max_time=10.0):
    """
    Attempts to find a small non-2-colorable hypergraph using edge pruning.
    """
    rng = random.Random(seed)
    
    # Choose n - smaller n gives denser hypergraphs
    n = rng.randint(8, 13)
    
    # Generate a starting set of edges
    all_edges = list(combinations(range(n), 5))
    
    # Strategy: Start with all edges and prune.
    edges = [list(e) for e in all_edges]
    rng.shuffle(edges)
    
    # Pruning loop with adaptive time allocation
    changed = True
    passes = 0
    start_time = time.time()
    last_progress = start_time
    
    while changed and time.time() - start_time < max_time:
        changed = False
        passes += 1
        current_edges = edges[:]
        rng.shuffle(current_edges)
        
        for e in current_edges:
            if time.time() - start_time > max_time:
                break
            
            # Try to remove e
            edges_without_e = [ed for ed in edges if ed != e]
            
            # Check if still non-2-colorable
            if _is_non_2colorable(n, edges_without_e):
                edges = edges_without_e
                changed = True
                last_progress = time.time()
    
    return n, len(edges), edges

def construct():
    best_edges = float('inf')
    best_result = None
    seed = 0
    
    # Fallback: Complete graph on 9 vertices (C(9,5) = 126 edges)
    fallback = {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    
    # Try a few quick attempts first with different n values
    for n in [8, 9, 10]:
        if time.time() > DEADLINE - 15:
            break
        all_edges = list(combinations(range(n), 5))
        edges = [list(e) for e in all_edges]
        if _is_non_2colorable(n, edges):
            if len(edges) < best_edges:
                best_edges = len(edges)
                best_result = {'n': n, 'edges': edges}
    
    # Main search loop
    while time.time() < DEADLINE - 5:
        seed += 1
        time_left = DEADLINE - time.time()
        if time_left < 8:
            break
        
        # Adaptive time allocation
        if time_left > 30:
            max_time_for_attempt = min(12.0, time_left * 0.4)
        else:
            max_time_for_attempt = min(8.0, time_left * 0.5)
        
        try:
            n, count, edges = _try_prune(seed, max_time_for_attempt)
            if count < best_edges:
                best_edges = count
                best_result = {'n': n, 'edges': edges}
        except Exception:
            pass
        
        # If we've found a good result, try to improve with targeted pruning
        if best_result and best_edges <= 60 and time.time() < DEADLINE - 10:
            # Try more aggressive pruning on the current best
            try:
                n_best = best_result['n']
                edges_best = best_result['edges'][:]
                rng = random.Random(seed + 1000)
                start_time = time.time()
                timeout = DEADLINE - 12 - time.time()
                
                changed = True
                while changed and time.time() - start_time < timeout:
                    changed = False
                    current = edges_best[:]
                    rng.shuffle(current)
                    for e in current:
                        if time.time() - start_time > timeout:
                            break
                        edges_without = [ed for ed in edges_best if ed != e]
                        if _is_non_2colorable(n_best, edges_without):
                            edges_best = edges_without
                            changed = True
                
                if len(edges_best) < best_edges:
                    best_edges = len(edges_best)
                    best_result = {'n': n_best, 'edges': edges_best}
            except Exception:
                pass
    
    if best_result is None:
        return fallback
        
    return best_result
