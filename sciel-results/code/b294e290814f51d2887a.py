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
    
    best_clique = []
    
    def bronk(R, P, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not P and not X:
            if R.bit_count() > len(best_clique):
                best_clique = [i + 1 for i in range(n) if (R >> i) & 1]
            return
        if not P:
            return
        
        # Choose pivot u from P ∪ X to maximize |P ∩ N(u)|
        union = P | X
        if union == 0:
            return
        pivot = 0
        max_count = -1
        temp = union
        while temp:
            lsb = temp & (-temp)
            u = lsb.bit_length() - 1
            temp ^= lsb
            count = (P & adj[u]).bit_count()
            if count > max_count:
                max_count = count
                pivot = u
            if max_count == P.bit_count():
                break
        
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
