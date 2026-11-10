import random
import time

def find_violation(n, adj):
    """
    Returns a bitmask of a regular induced subgraph of size >= 6, or None if valid.
    """
    # Check sizes from 6 up to n
    # It is often faster to check smaller sizes first if they are more likely to exist?
    # Actually, regular subgraphs of size 6 are very common in random graphs.
    # So checking size 6 first allows early termination.
    
    for size in range(6, n + 1):
        # Iterate through combinations of size 'size'
        # Using itertools.combinations is slower than bit manipulation for n=20?
        # 20 choose 6 is 38,760. 20 choose 10 is 184,756.
        # Bit manipulation over all masks is 2^20 = 1,048,576.
        # 1M iterations is fast. Let's stick to bit manipulation but filter by popcount.
        
        pass # We'll use the bit manipulation loop but break early on first violation.

def is_valid_graph_fast(n, adj):
    """
    Returns True if no subset of size >= 6 induces a regular subgraph.
    """
    for mask in range(1, 1 << n):
        m = mask.bit_count()
        if m < 6:
            continue
        
        lsb = mask & -mask
        u = lsb.bit_length() - 1
        target_deg = (adj[u] & mask).bit_count()
        
        rem_mask = mask ^ lsb
        valid = True
        while rem_mask:
            lsb2 = rem_mask & -rem_mask
            v = lsb2.bit_length() - 1
            deg_v = (adj[v] & mask).bit_count()
            if deg_v != target_deg:
                valid = False
                break
            rem_mask ^= lsb2
        
        if valid:
            return False
            
    return True

def construct():
    start_time = time.time()
    time_limit = 110  # Leave 10s for safety
    
    # Try n=21 first with local search
    n = 21
    found = None
    
    # Initial random graph
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Local search
    # We need a function to find a violation quickly.
    # The current is_valid_graph is O(2^n).
    # If we call it in a loop, it's slow.
    # We need to find *one* violation, not check all.
    
    def find_one_violation(adj, n):
        # Randomly shuffle the order of masks to avoid bias?
        # Or just check sequentially.
        # Sequential is deterministic.
        for mask in range(1, 1 << n):
            m = mask.bit_count()
            if m < 6:
                continue
            lsb = mask & -mask
            u = lsb.bit_length() - 1
            target_deg = (adj[u] & mask).bit_count()
            rem_mask = mask ^ lsb
            valid = True
            while rem_mask:
                lsb2 = rem_mask & -rem_mask
                v = lsb2.bit_length() - 1
                deg_v = (adj[v] & mask).bit_count()
                if deg_v != target_deg:
                    valid = False
                    break
                rem_mask ^= lsb2
            if valid:
                return mask
        return None

    # Local search loop
    while time.time() - start_time < time_limit - 5:
        viol_mask = find_one_violation(adj, n)
        if viol_mask is None:
            found = adj
            break
        
        # Pick a random edge in the violating subgraph to flip
        # Get vertices in viol_mask
        verts = []
        temp = viol_mask
        while temp:
            lsb = temp & -temp
            verts.append(lsb.bit_length() - 1)
            temp ^= lsb
        
        if len(verts) < 2:
            break # Should not happen for size >= 6
            
        # Pick two random vertices from the violating set
        u, v = random.sample(verts, 2)
        
        # Flip edge (u, v)
        if (adj[u] >> v) & 1:
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    if found:
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (found[i] >> j) & 1:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
        
    # If n=21 failed, try n=20
    n = 20
    found = None
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    while time.time() - start_time < time_limit:
        viol_mask = find_one_violation(adj, n)
        if viol_mask is None:
            found = adj
            break
        
        verts = []
        temp = viol_mask
        while temp:
            lsb = temp & -temp
            verts.append(lsb.bit_length() - 1)
            temp ^= lsb
            
        if len(verts) < 2:
            break
            
        u, v = random.sample(verts, 2)
        if (adj[u] >> v) & 1:
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    if found:
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (found[i] >> j) & 1:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
        
    # Fallback
    return {'n': 5, 'edges': []}
