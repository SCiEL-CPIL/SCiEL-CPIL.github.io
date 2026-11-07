"""
Combined program: Exact minimum cochromatic partition via Dynamic Programming.
For N=14, a DP over all subsets is efficient (O(3^N)) and guarantees the optimal solution.
This replaces the Branch-and-Bound approach with a more predictable and often faster method for small N.
"""
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

N = P['n']
adj = [0] * N
for u, v in P['edges']:
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

FULL = (1 << N) - 1

def is_clique(mask):
    if mask == 0:
        return True
    v = (mask & -mask).bit_length() - 1
    # Check if v is connected to all others in mask
    if mask & ~adj[v]:
        return False
    return is_clique(mask ^ (1 << v))

def is_independent(mask):
    if mask == 0:
        return True
    v = (mask & -mask).bit_length() - 1
    # Check if v is NOT connected to any other in mask
    if mask & adj[v]:
        return False
    return is_independent(mask ^ (1 << v))

def construct():
    start_time = time.time()
    deadline = 85.0 # Safety margin
    
    # 1. Precompute homogeneity for all masks
    # is_hom[mask] = 1 if mask is clique or independent, 0 otherwise
    # We also need to ensure mask is not 0 for the partition parts, 
    # but for DP logic, 0 is a valid state (empty set).
    # A single vertex is both a clique and an independent set.
    
    is_hom = [0] * (1 << N)
    is_hom[0] = 1 # Empty set is trivially homogeneous (though we won't use it as a part)
    
    for mask in range(1, 1 << N):
        if is_clique(mask):
            is_hom[mask] = 1
        elif is_independent(mask):
            is_hom[mask] = 1
            
    # 2. Dynamic Programming
    # dp[mask] = minimum number of parts to cover 'mask'
    # parent[mask] = the subset chosen as the last part in the optimal partition for 'mask'
    size = 1 << N
    dp = [10**9] * size
    parent = [-1] * size
    dp[0] = 0
    
    # Iterate through all masks
    # To optimize, we can iterate submasks, but for N=14, iterating all submasks for each mask is 3^N ~ 4.8M ops.
    # This is fast enough.
    
    for mask in range(1, size):
        if time.time() > deadline:
            # Fallback to greedy if we somehow run out of time (unlikely for N=14)
            return _greedy_fallback()
        
        best_val = 10**9
        best_sub = 0
        
        # Iterate over all submasks of mask
        # Standard submask iteration
        sub = mask
        while sub:
            if is_hom[sub]:
                prev = mask ^ sub
                val = dp[prev] + 1
                if val < best_val:
                    best_val = val
                    best_sub = sub
            sub = (sub - 1) & mask
            
        dp[mask] = best_val
        parent[mask] = best_sub

    # 3. Reconstruct the partition
    parts = []
    mask = FULL
    while mask:
        sub = parent[mask]
        if sub == 0:
            # This should not happen if dp is correct, but handle gracefully
            # Find any homogeneous subset
            found = False
            for i in range(N):
                if mask & (1 << i):
                    sub = 1 << i
                    found = True
                    break
            if not found:
                break
        parts.append(sub)
        mask ^= sub
        
    # Convert masks to lists of vertices
    result = []
    for m in parts:
        verts = []
        for i in range(N):
            if m & (1 << i):
                verts.append(i)
        result.append(verts)
        
    return result

def _greedy_fallback():
    """Fallback if DP times out (extremely unlikely for N=14)."""
    es = {tuple(e) for e in P['edges']}
    parts = []
    for v in range(N):
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es or (w, u) in es for i, u in enumerate(trial) for w in trial[i + 1:]]
            if all(hits) or not any(hits):
                part.append(v)
                break
        else:
            parts.append([v])
    return [list(part) for part in parts]
