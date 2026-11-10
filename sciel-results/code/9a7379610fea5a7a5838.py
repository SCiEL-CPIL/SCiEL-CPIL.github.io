import time
import random

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    def is_valid(n, adj):
        """
        Returns True if no regular induced subgraph of size >= 6 exists.
        Returns False otherwise.
        """
        # Iterate all masks
        # Optimization: Skip masks with < 6 bits
        for mask in range(1, 1 << n):
            k = mask.bit_count()
            if k < 6:
                continue
            
            # Check regularity
            m = mask
            first_deg = -1
            is_reg = True
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = (adj[v] & mask).bit_count()
                if first_deg == -1:
                    first_deg = deg
                elif deg != first_deg:
                    is_reg = False
                    break
                m -= lsb
            
            if is_reg:
                return False
        return True

    def get_one_bad_subset(n, adj):
        """
        Returns a mask of a regular induced subgraph of size >= 6, or None if valid.
        """
        for mask in range(1, 1 << n):
            k = mask.bit_count()
            if k < 6:
                continue
            
            m = mask
            first_deg = -1
            is_reg = True
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = (adj[v] & mask).bit_count()
                if first_deg == -1:
                    first_deg = deg
                elif deg != first_deg:
                    is_reg = False
                    break
                m -= lsb
            
            if is_reg:
                return mask
        return None

    # Try n from 22 down to 20
    for n in [22, 21, 20]:
        if time.time() > deadline - 10:
            break
            
        # Try multiple restarts
        for restart in range(50):
            if time.time() > deadline - 5:
                break
                
            # Initialize random graph
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search
            # We track the last bad subset to avoid immediate cycling
            last_bad = None
            stuck_count = 0
            
            for step in range(500):
                if time.time() > deadline - 2:
                    break
                
                bad_mask = get_one_bad_subset(n, adj)
                
                if bad_mask is None:
                    # Found a valid graph!
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if adj[u] & (1 << v):
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
                
                # If we found the same bad subset as before, we might be stuck
                if bad_mask == last_bad:
                    stuck_count += 1
                else:
                    stuck_count = 0
                    last_bad = bad_mask
                
                # If stuck, pick a random edge anywhere to escape
                if stuck_count > 5:
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u != v:
                        if adj[u] & (1 << v):
                            adj[u] ^= (1 << v)
                            adj[v] ^= (1 << u)
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                    continue
                
                # Pick a random edge inside the bad subset to flip
                verts = []
                m = bad_mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    verts.append(v)
                    m -= lsb
                
                if len(verts) < 2:
                    continue
                
                u, v = random.sample(verts, 2)
                
                # Flip edge (u, v)
                if adj[u] & (1 << v):
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

    # Fallback to a known valid small graph
    # n=5 empty graph is valid (no subset of size 6)
    return {'n': 5, 'edges': []}
