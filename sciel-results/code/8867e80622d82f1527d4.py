import random
import time

def _find_bad_subset(adj, n):
    """
    Find a subset of size >= 6 that induces a regular subgraph.
    Returns the mask of such a subset, or None if none exists.
    """
    # Quick random check
    for _ in range(2000):
        # Generate random mask with at least 6 bits
        # Simple method: pick random bits until we have 6, then add more
        # Or just pick random mask and check bit_count
        while True:
            mask = 0
            # Randomly select k vertices
            k = random.randint(6, n)
            verts = random.sample(range(n), k)
            for v in verts:
                mask |= (1 << v)
            break
        
        # Check if regular
        degs = []
        for u in range(n):
            if (mask >> u) & 1:
                degs.append((adj[u] & mask).bit_count())
        
        if len(set(degs)) == 1:
            return mask

    # Full deterministic scan
    # Iterate all masks
    # To speed up, we can iterate by size, but bit_count is fast
    # We can skip masks with < 6 bits
    # There are 2^n masks.
    # For n=21, 2^21 = 2,097,152.
    # Inner loop is 21 iterations.
    # Total ~44M operations.
    
    # Optimization: Precompute popcount? 
    # bit_count() is a C call, very fast.
    
    # We can also prune: if a mask is not regular, its subsets might be.
    # But we want to find ANY regular mask.
    
    # Let's just iterate.
    for mask in range(1, 1 << n):
        if mask.bit_count() < 6:
            continue
        
        # Check regularity
        # Optimize: check first two degrees, if different, skip rest?
        # No, all must be same.
        
        # Get list of vertices in mask
        # This is slow in Python.
        # Alternative:
        # degs = []
        # for u in range(n): ...
        
        # Faster:
        first_deg = -1
        is_reg = True
        for u in range(n):
            if (mask >> u) & 1:
                d = (adj[u] & mask).bit_count()
                if first_deg == -1:
                    first_deg = d
                elif d != first_deg:
                    is_reg = False
                    break
        if is_reg:
            return mask

    return None

def _extract_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Try n=21 first for a short duration
    # Then n=20
    
    for n in [21, 20]:
        if time.time() > deadline:
            break
            
        # If we are in the last n, use all remaining time
        # If n=21, limit to 30s
        phase_deadline = start_time + 30 if n == 21 else deadline
        
        # Local search
        while time.time() < phase_deadline - 2:
            # Random restart
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search loop
            for _ in range(500):  # Max steps per restart
                if time.time() > phase_deadline - 2:
                    break
                
                bad_mask = _find_bad_subset(adj, n)
                
                if bad_mask is None:
                    # Valid graph found
                    return {'n': n, 'edges': _extract_edges(adj, n)}
                
                # Repair: Flip an edge in the bad subset to break regularity
                # Pick a vertex u in bad_mask
                verts_in_mask = [i for i in range(n) if (bad_mask >> i) & 1]
                u = random.choice(verts_in_mask)
                
                # Pick a vertex v to flip with
                # Prefer vertices in the mask to disrupt degrees significantly
                # Or any vertex?
                # Flipping an edge inside the mask changes degrees of 2 vertices in mask by +/- 1.
                # This likely breaks regularity.
                v = random.choice(verts_in_mask)
                if u == v:
                    v = random.randint(0, n - 1)
                
                # Flip edge (u, v)
                if (adj[u] >> v) & 1:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                # If we flipped, the graph changed.
                # The next _find_bad_subset will find a (possibly different) bad subset.
            
            # If loop finishes, restart with new random graph
    
    # Fallback
    return {'n': 5, 'edges': []}
