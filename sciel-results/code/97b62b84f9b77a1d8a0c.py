import time
import random

def creates_c6(u, v, adj):
    """
    Check if adding edge (u, v) to the graph represented by adj (list of bitmasks)
    creates a C6.
    """
    nu = adj[u]
    nv = adj[v]
    
    # We are looking for a path u-a-b-c-d-v
    # a in N(u), d in N(v)
    # b in N(a), c in N(b), d in N(c)
    # All vertices u, v, a, b, c, d must be distinct.
    
    exclude_uv = ~(1 << u) & ~(1 << v)
    
    a_iter = nu
    while a_iter:
        a_bit = a_iter & -a_iter
        a = a_bit.bit_length() - 1
        a_iter ^= a_bit
        
        d_iter = nv
        while d_iter:
            d_bit = d_iter & -d_iter
            d = d_bit.bit_length() - 1
            d_iter ^= d_bit
            
            if a == d:
                continue
            
            # Find b in N(a) such that there exists c in N(b) \cap N(d)
            # b must not be u, v, a, d
            b_mask = adj[a] & exclude_uv & ~(1 << a) & ~(1 << d)
            
            while b_mask:
                b_bit = b_mask & -b_mask
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                
                # c must be in N(b) and N(d)
                # c must not be u, v, a, b, d
                c_mask = (adj[b] & adj[d]) & exclude_uv & ~(1 << a) & ~(1 << b) & ~(1 << d)
                
                if c_mask:
                    return True
    return False

def greedy_construct(seed):
    n = 40
    random.seed(seed)
    
    # Generate all possible edges
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
    
    random.shuffle(all_edges)
    
    adj = [0] * n
    edges = []
    
    for u, v in all_edges:
        # Check if edge already exists (shouldn't in this loop)
        # Check if adding creates C6
        if not creates_c6(u, v, adj):
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
            
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_edges = []
    best_len = 0
    
    seed = 0
    while time.time() < deadline:
        edges = greedy_construct(seed)
        if len(edges) > best_len:
            best_len = len(edges)
            best_edges = edges
            # Optional: print or log progress
            # print(f"Seed {seed}: {best_len} edges")
        seed += 1
        
        # Check time more frequently if we are close to deadline
        if seed % 100 == 0:
            if time.time() > deadline - 5:
                break
                
    return best_edges
