import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 512
    # Build adjacency list as bitmasks
    adj = [0] * n
    
    # Precompute gcds or just compute on the fly?
    # N=512, N^2/2 = ~131k pairs. gcd is fast.
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            l = a * b // gcd(a, b)
            if l <= 512:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Initial best clique: Divisors of 360 (size 24)
    # 360 = 2^3 * 3^2 * 5
    # Divisors count: 4 * 3 * 2 = 24
    # Let's just use [1..24] or a known good set?
    # Divisors of 360:
    divs_360 = []
    for i in range(1, 361):
        if 360 % i == 0:
            divs_360.append(i)
    # Convert to 0-indexed bitset
    best_clique_set = set(divs_360)
    best_clique_mask = 0
    for v in divs_360:
        best_clique_mask |= (1 << (v - 1))
    best_clique = divs_360
    best_size = len(best_clique)
    
    all_nodes = (1 << n) - 1
    
    def bronk(R, P, X):
        nonlocal best_clique, best_size
        
        # Time check
        if time.time() > deadline:
            return
            
        # Bound check: Can we beat the best?
        if R.bit_count() + P.bit_count() <= best_size:
            return
            
        if not P:
            if not X:
                # Found a maximal clique
                if R.bit_count() > best_size:
                    best_clique = [i + 1 for i in range(n) if (R >> i) & 1]
                    best_size = len(best_clique)
            return
            
        if not P:
            return
            
        # Pivot selection
        # Choose u in P | X to maximize |P & adj[u]|
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
                
        # Iterate over candidates P \ N(pivot)
        candidates = P & ~adj[pivot]
        while candidates:
            lsb = candidates & (-candidates)
            v = lsb.bit_length() - 1
            candidates ^= lsb
            
            bronk(R | lsb, P & adj[v], X & adj[v])
            
            P ^= lsb
            X |= lsb
            
            # Optional: If P is empty, break early? No, X might not be empty.
            # But if P is empty, the loop ends naturally.
            
    bronk(0, all_nodes, 0)
    
    if not best_clique:
        return [1]
        
    return best_clique
