import time

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 40
    # Use bitmasks for adjacency
    adj = [0] * n  # adj[i] is bitmask of neighbors
    
    def has_c6_with_edge(u, v):
        """Check if adding edge (u,v) creates a C6 using current adj."""
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        
        # For each a in nu, d in nv, check if path a-b-c-d exists
        a = nu
        while a:
            la = a & (-a)
            ai = la.bit_length() - 1
            a ^= la
            
            d = nv
            forbidden = (1 << u) | (1 << v) | (1 << ai)
            while d:
                ld = d & (-d)
                di = ld.bit_length() - 1
                d ^= ld
                
                if di == ai:
                    continue
                forbidden_d = forbidden | (1 << di)
                
                # b in N(ai), c in N(di), b~c, b,c not in forbidden_d
                na = adj[ai] & ~forbidden_d
                nd = adj[di] & ~forbidden_d
                
                # Check if any b in na has a neighbor c in nd
                b = na
                while b:
                    lb = b & (-b)
                    bi = lb.bit_length() - 1
                    b ^= lb
                    
                    if adj[bi] & nd:
                        return True
        
        return False
    
    def count_edges():
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Greedy: try to add edges in random order
    import random
    random.seed(42)
    
    best_count = 0
    best_adj = None
    
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        
        adj = [0] * n
        shuffled = all_pairs[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            if time.time() > deadline - 3:
                break
            if not (adj[u] >> v) & 1:
                if not has_c6_with_edge(u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Second pass
        for u, v in shuffled:
            if time.time() > deadline - 3:
                break
            if not (adj[u] >> v) & 1:
                if not has_c6_with_edge(u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        c = count_edges()
        if c > best_count:
            best_count = c
            best_adj = adj[:]
        
        if best_count >= 105:
            break
    
    if best_adj is None:
        # Fallback: 8 K5
        edges = []
        for k in range(8):
            edges += [[5*k+a, 5*k+b] for a in range(5) for b in range(a+1, 5)]
        return edges
    
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
