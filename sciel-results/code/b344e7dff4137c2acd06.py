import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 512
    # Build adjacency: node i (0-indexed) represents value i+1
    # adj[i] is a bitmask of neighbors
    adj = [0] * n
    
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            l = a * b // gcd(a, b)
            if l <= 512:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Compute degree for each node (number of neighbors)
    degrees = [adj[i].bit_count() for i in range(n)]
    
    best_clique = []
    best_size = 0
    # Upper bound: greedy coloring gives an upper bound on clique size in P
    # But for simplicity, we use a looser bound initially
    
    def bronk(R, P, X):
        nonlocal best_clique, best_size
        if time.time() > deadline:
            return
        if not P and not X:
            sz = R.bit_count()
            if sz > best_size:
                best_size = sz
                best_clique = [i + 1 for i in range(n) if (R >> i) & 1]
            return
        if not P:
            return
        
        # Upper bound: if |R| + |P| <= best_size, we can't improve
        if R.bit_count() + P.bit_count() <= best_size:
            return
        
        # Choose pivot u from P ∪ X to maximize |P ∩ N(u)|
        union = P | X
        if union == 0:
            return
        
        # Fast pivot: pick node with max (P & adj[u]).bit_count()
        # Try a few candidates from high-degree nodes first
        pivot = -1
        max_count = -1
        temp = union
        # Limit pivot search to avoid spending too much time
        pivot_search_limit = 50
        search_count = 0
        while temp and search_count < pivot_search_limit:
            lsb = temp & (-temp)
            u = lsb.bit_length() - 1
            temp ^= lsb
            count = (P & adj[u]).bit_count()
            if count > max_count:
                max_count = count
                pivot = u
                if count == P.bit_count():
                    break
            search_count += 1
        
        if pivot == -1:
            # Fallback: pick first node in union
            lsb = union & (-union)
            pivot = lsb.bit_length() - 1
        
        # Iterate over P \ N(pivot)
        candidates = P & ~adj[pivot]
        while candidates:
            lsb = candidates & (-candidates)
            v = lsb.bit_length() - 1
            candidates ^= lsb
            
            bronk(R | lsb, P & adj[v], X & adj[v])
            P ^= lsb
            X |= lsb
    
    all_nodes = (1 << n) - 1
    bronk(0, all_nodes, 0)
    
    if not best_clique:
        return [1]
    
    return best_clique
