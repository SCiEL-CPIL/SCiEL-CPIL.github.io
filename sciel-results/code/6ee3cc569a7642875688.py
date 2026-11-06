import random
import time

def construct():
    n = 20
    deadline = time.time() + 110  # Leave 10s margin for 120s limit
    
    # Precompute all subsets of size 6
    subsets_6 = []
    for mask in range(1 << n):
        if bin(mask).count('1') == 6:
            subsets_6.append(mask)
            
    # Helper to check if a subset is regular
    def is_regular(mask, neighbors):
        # Get vertices in mask
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        
        if not verts:
            return True # Empty is regular (0-regular) but we care about >= 6
        
        degs = set()
        for v in verts:
            d = bin(neighbors[v] & mask).count('1')
            degs.add(d)
            if len(degs) > 1:
                return False
        return True

    # Helper to count bad subsets of size 6
    def get_bad_6(neighbors):
        bad = []
        for mask in subsets_6:
            if is_regular(mask, neighbors):
                bad.append(mask)
        return bad

    # Helper to check full validity
    def is_valid_graph(neighbors):
        # Check all subsets of size >= 6
        # We can iterate through all masks with popcount >= 6
        # To save time, we can just check if any is regular
        # Optimization: if we know size 6 are clear, check 7, etc.
        # But for final check, we must be sure.
        for mask in range(1 << n):
            pc = bin(mask).count('1')
            if pc < 6:
                continue
            if is_regular(mask, neighbors):
                return False
        return True

    best_score = -1
    best_graph = None
    
    # We will try multiple restarts
    while time.time() < deadline:
        # Initialize random graph
        neighbors = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    neighbors[i] |= (1 << j)
                    neighbors[j] |= (1 << i)
        
        # Local search
        for step in range(500): # Max steps per restart
            if time.time() > deadline - 5:
                break
                
            bad_6 = get_bad_6(neighbors)
            if not bad_6:
                # Check full validity
                if is_valid_graph(neighbors):
                    # Construct edge list
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if neighbors[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                # If not valid, we have bad subsets of size > 6.
                # We can try to fix them by picking a random bad subset of size > 6
                # For simplicity, we can just continue random walk or restart
                # Let's find one bad subset > 6
                bad_large = None
                # Sample some large subsets to find a bad one quickly
                for _ in range(100):
                    # Random subset of size 7 to 10
                    k = random.randint(7, 10)
                    verts = random.sample(range(n), k)
                    mask = 0
                    for v in verts:
                        mask |= (1 << v)
                    if is_regular(mask, neighbors):
                        bad_large = mask
                        break
                
                if bad_large:
                    # Pick random edge in bad_large to flip
                    verts = []
                    m = bad_large
                    while m:
                        lsb = m & -m
                        v = lsb.bit_length() - 1
                        verts.append(v)
                        m ^= lsb
                    if len(verts) >= 2:
                        u, v = random.sample(verts, 2)
                        # Flip edge u-v
                        if neighbors[u] & (1 << v):
                            neighbors[u] ^= (1 << v)
                            neighbors[v] ^= (1 << u)
                        else:
                            neighbors[u] |= (1 << v)
                            neighbors[v] |= (1 << u)
                    continue
                else:
                    # Couldn't find a bad large subset in sample, maybe it's valid?
                    # Re-check
                    if is_valid_graph(neighbors):
                        edges = []
                        for i in range(n):
                            for j in range(i+1, n):
                                if neighbors[i] & (1 << j):
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}
            
            # Pick a random bad subset of size 6
            S = random.choice(bad_6)
            # Get vertices in S
            verts = []
            m = S
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
            
            # Pick random pair in S to flip
            if len(verts) >= 2:
                u, v = random.sample(verts, 2)
                # Flip edge u-v
                if neighbors[u] & (1 << v):
                    neighbors[u] ^= (1 << v)
                    neighbors[v] ^= (1 << u)
                else:
                    neighbors[u] |= (1 << v)
                    neighbors[v] |= (1 << u)

    # If we run out of time, return the best found or a default
    # Since we didn't track best, we return a known small valid graph if we fail
    # But we should try to return something valid if possible.
    # The empty graph on 5 is valid.
    return {'n': 5, 'edges': []}
