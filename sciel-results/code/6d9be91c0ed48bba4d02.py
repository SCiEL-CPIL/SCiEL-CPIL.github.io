import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # All numbers in [1,255] that share a factor with 255
    candidates = [i for i in range(1, 256) if math.gcd(i, 255) > 1]
    
    # Build adjacency: adj[i] is a bitmask of neighbors
    n = len(candidates)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Bron-Kerbosch with pivoting
    best_clique = [0] * n  # bitmask of best clique found
    best_size = 0
    
    def popcount(x):
        return bin(x).count("1")
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique, best_size
        if time.time() > DEADLINE:
            return
        if P == 0 and X == 0:
            sz = popcount(R)
            if sz > best_size:
                best_clique = R
                best_size = sz
            return
        # Choose pivot u from P | X to maximize |P & adj[u]|
        U = P | X
        if U == 0:
            return
        best_u = -1
        best_count = -1
        temp = U
        while temp:
            lb = temp & (-temp)
            u = lb.bit_length() - 1
            temp ^= lb
            cnt = popcount(P & adj[u])
            if cnt > best_count:
                best_count = cnt
                best_u = u
        # Iterate over P \ adj[best_u]
        candidates_to_try = P & ~(adj[best_u])
        temp = candidates_to_try
        while temp:
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            temp ^= lb
            bron_kerbosch(R | lb, P & adj[v], X & adj[v])
            P ^= lb
            X |= lb
            if time.time() > DEADLINE:
                return
    
    bron_kerbosch(0, (1 << n) - 1, 0)
    
    # Extract clique
    result = []
    for i in range(n):
        if best_clique & (1 << i):
            result.append(candidates[i])
    
    # Ensure 255 is in the result (it should be, but verify)
    if 255 not in result:
        # Fallback: use multiples of 5
        result = list(range(5, 256, 5))
    
    return result
