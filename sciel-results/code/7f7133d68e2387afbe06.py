"""Construct a K4-free graph with no independent 6-set, maximizing n."""
import time
import random
import itertools

_DEADLINE = 150.0
_START_TIME = time.time()

def _time_left():
    return _DEADLINE - 10.0 - (time.time() - _START_TIME)

def _build_adj_from_shifts(n, shifts):
    """Build adjacency list from a set of shifts."""
    adj = [0] * n
    for i in range(n):
        for s in shifts:
            j = (i + s) % n
            if i < j:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    return adj

def _is_k4_free_circulant(n, shift_mask):
    """
    Check if a circulant graph defined by shift_mask is K4-free.
    shift_mask is an integer where bit i is 1 if shift i is in S.
    We check all patterns {0, a, b, c}.
    """
    # Precomputed patterns for this n? No, n varies.
    # But we can generate them on the fly for n=35/36.
    # For efficiency, we assume n is fixed during the call.
    
    # Get the set of active shifts U = S U -S
    # shift_mask has bits 1..n//2.
    # U contains s if bit s is set, or bit n-s is set?
    # Actually, shift_mask represents S. U = S U -S.
    # In the pattern check, we need to know if a difference d is in U.
    # d is in U if (d % n) is in S or (-d % n) is in S.
    
    # Let's create a boolean array is_in_U of size n
    is_in_U = [False] * n
    for s in range(1, n // 2 + 1):
        if (shift_mask >> s) & 1:
            is_in_U[s] = True
            is_in_U[n - s] = True
    
    # Check all patterns {0, a, b, c} with 0 < a < b < c < n
    # Differences: a, b, c, b-a, c-a, c-b
    # All must be in U
    
    # To speed up, we can iterate a, b, c
    # But 35^3 is 42k. 2^17 * 42k is too much.
    
    # Better: Precompute the "required differences" for each pattern as a bitmask?
    # No, we just check the 6 differences.
    
    # Optimization: If any of a, b, c is not in U, skip.
    # This prunes a lot.
    
    for a in range(1, n):
        if not is_in_U[a]: continue
        for b in range(a + 1, n):
            if not is_in_U[b]: continue
            if not is_in_U[b - a]: continue
            for c in range(b + 1, n):
                if not is_in_U[c]: continue
                if not is_in_U[c - a]: continue
                if not is_in_U[c - b]: continue
                return False # Found a K4
    return True

def _find_ind6_exact(adj, n):
    """Find an independent set of size 6, or None."""
    # Complement graph
    comp_adj = [0] * n
    for i in range(n):
        comp_adj[i] = ((1 << n) - 1) ^ (1 << i) ^ adj[i]
    
    # Bron-Kerbosch for K6 in comp_adj
    # Pivot selection to speed up
    best = [None]
    
    def search(current, candidates, pivot):
        if len(current) == 6:
            best[0] = tuple(current)
            return True
        if not candidates:
            return False
        if len(current) + bin(candidates).count('1') < 6:
            return False
            
        # Pivot
        p = -1
        max_common = 0
        cand_list = []
        for i in range(n):
            if (candidates >> i) & 1:
                cand_list.append(i)
        
        if not cand_list:
            return False
            
        # Simple pivot: pick first candidate
        # Better: pick one with max common neighbors in candidates
        p = cand_list[0]
        for v in cand_list:
            common = comp_adj[v] & candidates
            cnt = bin(common).count('1')
            if cnt > max_common:
                max_common = cnt
                p = v
        
        # Iterate over candidates NOT in N(p)
        non_neighbors = candidates & ~comp_adj[p]
        
        for i in range(n):
            if (non_neighbors >> i) & 1:
                v = i
                current.append(v)
                if search(current, candidates & comp_adj[v], -1):
                    return True
                current.pop()
                candidates &= ~(1 << v)
                
        return False

    if search([], (1 << n) - 1, -1):
        return best[0]
    return None

def _edges_from_adj(adj, n):
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
    return edges

def construct():
    # Try n=36 first
    for n in [36, 35]:
        if _time_left() < 10:
            break
            
        # Search circulant graphs
        max_shift = n // 2
        # Randomize the order of search or iterate all?
        # Iterate all is 2^18 for 36.
        
        # To be safe, use a time-limited random search first, then exhaustive if time permits?
        # Or just exhaustive if fast enough.
        
        # Let's try a heuristic: start with a random mask, improve?
        # No, let's just iterate.
        
        found = False
        # Use a list of candidates to check
        # For n=36, 2^18 = 262144.
        # For n=35, 2^17 = 131072.
        
        # We will iterate through all masks.
        # To speed up, we can skip masks that are clearly bad?
        
        for mask in range(1 << max_shift):
            if _time_left() < 5:
                break
                
            if not _is_k4_free_circulant(n, mask):
                continue
                
            # If K4-free, check Ind 6
            shifts = [i for i in range(1, max_shift + 1) if (mask >> i) & 1]
            adj = _build_adj_from_shifts(n, shifts)
            
            if _find_ind6_exact(adj, n) is None:
                edges = _edges_from_adj(adj, n)
                return {'n': n, 'edges': edges}
                
    # Fallback: Local Search for n=35
    # ... (implement a simple local search if circulant fails)
    
    # For now, return a known small graph if all else fails
    # Paley 17
    n = 17
    squares = {x * x % 17 for x in range(1, 17)}
    edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    return {'n': n, 'edges': edges}
