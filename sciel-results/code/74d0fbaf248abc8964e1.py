import time
import random

def _is_regular_mask(adj, mask):
    """Check if induced subgraph on mask is regular."""
    cnt = mask.bit_count()
    if cnt < 6:
        return False
    first_deg = -1
    m = mask
    while m:
        lb = m & (-m)
        i = lb.bit_length() - 1
        deg = (adj[i] & mask).bit_count()
        if first_deg == -1:
            first_deg = deg
        elif deg != first_deg:
            return False
        m ^= lb
    return True

def _find_violation(adj, n):
    """Find a mask of size >= 6 that is regular. Returns None if valid."""
    # Iterate masks
    # Optimization: Check smaller masks first? No, any size >= 6.
    # Random order might find one faster?
    # Let's just iterate.
    for mask in range(1, 1 << n):
        if mask.bit_count() >= 6:
            if _is_regular_mask(adj, mask):
                return mask
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_n = 5
    best_edges = []
    
    # Try to build n=20
    target_n = 20
    random.seed(12345)
    
    # Initialize random graph
    adj = [0] * target_n
    edges = []
    for i in range(target_n):
        for j in range(i + 1, target_n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append([i, j])
                
    # Local search
    for step in range(500):
        if time.time() > deadline:
            break
            
        violation = _find_violation(adj, target_n)
        if violation is None:
            # Valid graph found
            best_n = target_n
            # Reconstruct edges from adj
            best_edges = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if adj[i] & (1 << j):
                        best_edges.append([i, j])
            return {'n': best_n, 'edges': best_edges}
            
        # Fix violation
        # Get vertices in violation
        verts = []
        m = violation
        while m:
            lb = m & (-m)
            verts.append(lb.bit_length() - 1)
            m ^= lb
            
        # Pick a random vertex in the violation and flip edges to other vertices in violation
        v = random.choice(verts)
        # Flip edge (v, u) for a random u in verts
        u = random.choice([x for x in verts if x != v])
        
        # Flip
        if adj[v] & (1 << u):
            adj[v] &= ~(1 << u)
            adj[u] &= ~(1 << v)
            # Update edges list? 
            # It's easier to just reconstruct at the end.
        else:
            adj[v] |= (1 << u)
            adj[u] |= (1 << v)
            
        # Occasionally flip an edge outside the violation to escape local minima
        if random.random() < 0.1:
            i = random.randint(0, target_n - 1)
            j = random.randint(0, target_n - 1)
            if i != j:
                if adj[i] & (1 << j):
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)

    # If n=20 failed, try n=19
    if best_n < 19:
        target_n = 19
        random.seed(54321)
        adj = [0] * target_n
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        for step in range(200):
            if time.time() > deadline:
                break
            violation = _find_violation(adj, target_n)
            if violation is None:
                best_n = target_n
                best_edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if adj[i] & (1 << j):
                            best_edges.append([i, j])
                return {'n': best_n, 'edges': best_edges}
            
            verts = []
            m = violation
            while m:
                lb = m & (-m)
                verts.append(lb.bit_length() - 1)
                m ^= lb
                
            v = random.choice(verts)
            u = random.choice([x for x in verts if x != v])
            if adj[v] & (1 << u):
                adj[v] &= ~(1 << u)
                adj[u] &= ~(1 << v)
            else:
                adj[v] |= (1 << u)
                adj[u] |= (1 << v)

    # Fallback: return best found or a small valid graph
    if best_n < 6:
        # Empty graph on 5 vertices is valid
        best_n = 5
        best_edges = []
        
    # Reconstruct edges for best_n if we found one
    # (Note: if we didn't return inside the loop, we might have a valid adj from the last attempt? 
    # No, we only set best_n if valid. But we didn't store edges. 
    # We should store edges when we find a valid one.)
    
    # Actually, the code above returns immediately if valid is found.
    # So if we reach here, we failed to find n=20 or n=19.
    # We should try n=18?
    # For now, return n=5 to be safe, or try n=18 quickly.
    
    # Let's try n=18
    if best_n < 18 and time.time() < deadline - 10:
        target_n = 18
        random.seed(11111)
        adj = [0] * target_n
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        for step in range(100):
            if time.time() > deadline - 5:
                break
            violation = _find_violation(adj, target_n)
            if violation is None:
                best_n = target_n
                best_edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if adj[i] & (1 << j):
                            best_edges.append([i, j])
                return {'n': best_n, 'edges': best_edges}
            
            verts = []
            m = violation
            while m:
                lb = m & (-m)
                verts.append(lb.bit_length() - 1)
                m ^= lb
                
            v = random.choice(verts)
            u = random.choice([x for x in verts if x != v])
            if adj[v] & (1 << u):
                adj[v] &= ~(1 << u)
                adj[u] &= ~(1 << v)
            else:
                adj[v] |= (1 << u)
                adj[u] |= (1 << v)

    return {'n': best_n, 'edges': best_edges}
