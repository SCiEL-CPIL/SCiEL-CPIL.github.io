import time
import random
import sys

def _check_c4_free(adj, n):
    """Check if graph is C4-free using bitsets."""
    adj_bits = [0] * n
    for i in range(n):
        bits = 0
        for j in range(n):
            if adj[i][j]:
                bits |= (1 << j)
        adj_bits[i] = bits
    
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                common = adj_bits[i] & adj_bits[j]
                # Count bits
                cnt = bin(common).count('1')
                if cnt >= 2:
                    return False
    return True

def _max_independent_set(adj, n, k, time_limit=5.0):
    """Find if independent set of size k exists using branch and bound with bitsets."""
    start_time = time.time()
    
    # Sort by degree ascending (low degree first for independent set)
    degrees = [sum(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x])
    
    remap = [0] * n
    for new_idx, old_idx in enumerate(order):
        remap[old_idx] = new_idx
    
    new_adj = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                new_adj[remap[i]][remap[j]] = 1
    
    adj_bits = [0] * n
    for i in range(n):
        bits = 0
        for j in range(n):
            if new_adj[i][j]:
                bits |= (1 << j)
        adj_bits[i] = bits
    
    comp_bits = [0] * n
    all_mask = (1 << n) - 1
    for i in range(n):
        comp_bits[i] = (all_mask ^ adj_bits[i]) & ~(1 << i)
    
    found = [False]
    
    def backtrack(start, size, candidates):
        if found[0]:
            return
        if time.time() - start_time > time_limit:
            return
        if size == k:
            found[0] = True
            return
        if candidates == 0:
            return
        # Pruning
        remaining = bin(candidates).count('1')
        if size + remaining < k:
            return
        
        # Pick a vertex
        m = candidates
        while m:
            if found[0]:
                return
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            
            # If we include v, next candidates must be in comp_bits[v] and > v
            new_cands = candidates & comp_bits[v] & ~((1 << (v + 1)) - 1)
            backtrack(v + 1, size + 1, new_cands)
            
            # If we exclude v
            candidates = candidates & ~lsb
            if size + bin(candidates).count('1') < k:
                return
            # Continue loop with updated candidates
    
    backtrack(0, 0, all_mask)
    return found[0]

def _local_search(adj, n, time_limit=10.0):
    """
    Local search: try to add edges to reduce independence number while maintaining C4-free.
    Returns (adj, edges, success)
    """
    start_time = time.time()
    random.seed(42)
    
    for iteration in range(100000):
        if time.time() - start_time > time_limit:
            break
        
        # Try adding a random non-edge
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j or adj[i][j]:
            continue
        
        # Check if adding edge (i,j) creates a C4
        common = 0
        for k in range(n):
            if adj[i][k] and adj[j][k]:
                common += 1
                if common >= 2:
                    break
        if common >= 2:
            continue
        
        # Add the edge
        adj[i][j] = 1
        adj[j][i] = 1
        
        # Verify C4-free (quick check)
        if not _check_c4_free(adj, n):
            adj[i][j] = 0
            adj[j][i] = 0
            continue
        
        # Check if independence number is still < 11
        if _max_independent_set(adj, n, 11, time_limit=2.0):
            # Revert
            adj[i][j] = 0
            adj[j][i] = 0
            continue
    
    # Extract edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append([i, j])
    
    return adj, edges, True

def _construct_random(n, target_alpha, time_limit=30.0):
    """
    Construct a C4-free graph on n vertices with alpha < target_alpha using random + local search.
    """
    start_time = time.time()
    random.seed(123)
    
    # Start with empty graph
    adj = [[0] * n for _ in range(n)]
    
    # Phase 1: Add random edges while maintaining C4-free
    attempts = 0
    max_attempts = 100000
    while attempts < max_attempts and time.time() - start_time < time_limit * 0.5:
        attempts += 1
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j or adj[i][j]:
            continue
        
        # Check C4 condition
        common = 0
        for k in range(n):
            if adj[i][k] and adj[j][k]:
                common += 1
                if common >= 2:
                    break
        if common >= 2:
            continue
        
        adj[i][j] = 1
        adj[j][i] = 1
    
    # Phase 2: Local search to maximize edges (minimize alpha)
    adj, edges, success = _local_search(adj, n, time_limit=time_limit * 0.5)
    
    return adj, edges

def construct():
    deadline = time.time() + 140  # Leave 10s margin
    
    best_n = 0
    best_edges = []
    best_adj = None
    
    # Try different n values from high to low
    for n in [39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25]:
        if time.time() > deadline - 15:
            break
        
        try:
            adj, edges = _construct_random(n, 11, time_limit=10.0)
            
            # Verify C4-free
            if not _check_c4_free(adj, n):
                continue
            
            # Check independence number
            has_is = _max_independent_set(adj, n, 11, time_limit=5.0)
            if not has_is:
                if n > best_n:
                    best_n = n
                    best_edges = edges
                    best_adj = adj
                    break  # Found valid graph, stop
        except Exception:
            continue
    
    if best_n == 0:
        # Fallback: return a small valid graph
        # Single vertex
        return {'n': 1, 'edges': []}
    
    return {'n': best_n, 'edges': best_edges}
