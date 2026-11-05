import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Fallback: Paley graph on 17
    squares = {x * x % 17 for x in range(1, 17)}
    best_n = 17
    best_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    
    # Try to find larger graphs
    for n in range(35, 16, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this n
        found = False
        attempts = 0
        while time.time() < deadline - 10.0 and not found:
            attempts += 1
            # Random initialization
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            # Repair loop
            success = repair_graph(n, adj, deadline)
            if success:
                # Extract edges
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
                
    return {'n': best_n, 'edges': best_edges}

def repair_graph(n, adj, deadline):
    mask = (1 << n) - 1
    
    def has_k4():
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    c = adj[u] & adj[v]
                    # Check if c contains an edge
                    # Iterate bits in c
                    cc = c
                    while cc:
                        lsb = cc & -cc
                        w = lsb.bit_length() - 1
                        cc ^= lsb
                        if adj[w] & c:
                            return (u, v, w)
        return None

    def has_i6():
        # Check for K6 in complement
        # comp_adj[i] = mask ^ adj[i] ^ (1 << i)
        comp_adj = [mask ^ adj[i] ^ (1 << i) for i in range(n)]
        
        # Helper to check if a set of vertices (bitset) contains a K4 in comp_adj
        def has_k4_in_set(bitset):
            # Iterate pairs in bitset
            b = bitset
            while b:
                lsb1 = b & -b
                u = lsb1.bit_length() - 1
                b ^= lsb1
                # Common neighbors in bitset
                c = comp_adj[u] & bitset & b # neighbors in remaining bits
                cc = c
                while cc:
                    lsb2 = cc & -cc
                    v = lsb2.bit_length() - 1
                    cc ^= lsb2
                    # Check if u,v have a common neighbor in bitset that is connected to both
                    # Actually, we need a K4. u,v are connected.
                    # We need 2 more vertices w,x in bitset connected to u,v and each other.
                    # c contains neighbors of u in bitset (excluding u).
                    # We found v in c.
                    # Now check if v has another neighbor w in c (excluding v) that is connected to u?
                    # All in c are connected to u.
                    # So we need an edge (w,x) in c? No, we need a triangle in c?
                    # If c contains a K3, then u + K3 = K4.
                    # So check if c contains a triangle.
                    # Simpler: just check if c contains an edge (w,x) such that w,x are connected.
                    # Wait, if w,x are in c, they are connected to u.
                    # If w,x are connected to each other, then u,w,x is a triangle.
                    # We need K4. So we need u,v,w,x.
                    # u,v are connected.
                    # w,x must be in c (connected to u).
                    # w,x must be connected to v?
                    # And w,x connected to each other?
                    # This is getting complicated.
                    # Let's just check if the subgraph induced by bitset has a K4.
                    # For n=35, bitset size is small.
                    pass
            return None

        # Simplified: Just find any K6 in comp
        # Iterate 6-tuples? Too slow.
        # Use the same logic as has_k4 but for K6.
        # For each edge (u,v) in comp, c = comp_adj[u] & comp_adj[v].
        # If c contains a K4 in comp, then we have a K6.
        for u in range(n):
            for v in range(u + 1, n):
                if comp_adj[u] & (1 << v):
                    c = comp_adj[u] & comp_adj[v]
                    # Check if c contains K4 in comp
                    # This is expensive.
                    pass
        return None

    # This implementation is too slow/complex for the helper.
    # Let's use a simpler repair: just flip random edges in violation.
    # We will just use the has_k4 check and a simple has_i6 check.
    
    # Re-implementing has_i6 with a simpler brute force for small n?
    # No, let's just use the K4 check on complement for K6?
    # If complement has K6, it has K4.
    # So if we find K4 in complement, we can break it.
    # Breaking K4 in complement breaks I4 in original.
    # But we need to break I6.
    # If we break all I4's, we might not have I6.
    # So minimizing I4's is a good proxy.
    
    # Let's just repair K4's and I4's.
    # If no K4 and no I4, then no K6 (since K6 contains K4) and no I6 (since I6 contains I4).
    # Wait, K6 contains K4? Yes.
    # I6 contains I4? Yes.
    # So if we ensure no K4 and no I4, we are safe.
    # This is a stronger condition, but easier to check.
    # R(4,4) = 18. So for n > 18, we MUST have K4 or I4.
    # So we cannot have no K4 and no I4 for n > 18.
    # So this strategy fails for n > 18.
    
    # Back to checking K4 and I6.
    # I will implement a basic repair that finds a K4 and flips an edge.
    # And finds an I6 by checking complement for K6.
    
    # Given time constraints, I will just return the Paley graph if I can't implement the fast checker quickly.
    # But I can implement the K4 checker.
    # For I6, I will check if complement has K6.
    # I will approximate I6 check by checking if complement has K4? No.
    
    # Let's just do a random search for n=35 with the K4 checker and a slow I6 checker.
    # If it's too slow, it will timeout and return 17.
    
    return False

# The above logic is flawed. I need a working solution.
# I will provide a solution that uses the Paley graph for 17, and attempts a simple local search for higher n.
# If the local search is too slow, it will fall back.

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Fallback
    squares = {x * x % 17 for x in range(1, 17)}
    best_n = 17
    best_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    
    # Try n=35
    n = 35
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Simple repair
    for _ in range(10000):
        if time.time() > deadline:
            break
        # Check K4
        violation = None
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    c = adj[u] & adj[v]
                    cc = c
                    while cc:
                        lsb = cc & -cc
                        w = lsb.bit_length() - 1
                        cc ^= lsb
                        if adj[w] & c:
                            violation = (u, v, w)
                            break
            if violation:
                break
        if violation:
            # Flip edge u-v
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            continue
            
        # Check I6 (K6 in complement)
        # This is expensive. Skip for now or use a slower check.
        # If we skip, we might return a graph with I6.
        # We must check I6.
        
        # Slow I6 check:
        comp_adj = [((1 << n) - 1) ^ adj[i] ^ (1 << i) for i in range(n)]
        i6_found = False
        # Check all 6-tuples? 35 choose 6 is 1.6 million.
        # 1.6 million * 15 edge checks = 24 million ops.
        # In Python, this is ~1-2 seconds per check.
        # Too slow for 10000 iterations.
        
        # We will just check I6 every 100 iterations.
        if _ % 100 == 0:
            # Brute force I6 check
            for i in range(n):
                for j in range(i+1, n):
                    if comp_adj[i] & (1 << j):
                        for k in range(j+1, n):
                            if comp_adj[j] & (1 << k):
                                for l in range(k+1, n):
                                    if comp_adj[k] & (1 << l):
                                        for m in range(l+1, n):
                                            if comp_adj[l] & (1 << m):
                                                for p in range(m+1, n):
                                                    if comp_adj[m] & (1 << p) and comp_adj[l] & (1 << p) and comp_adj[k] & (1 << p) and comp_adj[j] & (1 << p) and comp_adj[i] & (1 << p):
                                                        i6_found = True
                                                        break
                                                if i6_found: break
                                            if i6_found: break
                                        if i6_found: break
                                    if i6_found: break
                                if i6_found: break
                            if i6_found: break
                        if i6_found: break
                    if i6_found: break
                if i6_found: break
                
            if i6_found:
                # Flip a random edge in the I6? We didn't save it.
                # Just flip a random edge.
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
            else:
                # Valid!
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
                
    return {'n': best_n, 'edges': best_edges}
