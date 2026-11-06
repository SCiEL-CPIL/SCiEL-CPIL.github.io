import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # 255 = 3 * 5 * 17
    # Candidates: numbers in [1,255] that share a factor with 255
    candidates = [i for i in range(1, 256) if math.gcd(i, 255) > 1]
    if 255 not in candidates:
        candidates.append(255)
    
    n = len(candidates)
    idx = {v: i for i, v in enumerate(candidates)}
    idx_255 = idx[255]
    
    # Build adjacency as bitmasks (non-coprime graph)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    def popcount(x):
        return bin(x).count("1")
        
    # Initial best: all multiples of 3
    init_mask = 0
    for i in range(n):
        if candidates[i] % 3 == 0:
            init_mask |= (1 << i)
    best_size = popcount(init_mask)
    best_mask = init_mask
    
    # If 255 is not in init_mask (it is, but just in case), ensure it is
    if not (init_mask & (1 << idx_255)):
        # This shouldn't happen since 255 is divisible by 3, 5, 17
        pass

    # Bron-Kerbosch with pivot
    def bron_kerbosch(R, P, X):
        nonlocal best_size, best_mask
        if time.time() - start_time > DEADLINE:
            return
        
        r_size = popcount(R)
        if r_size > best_size:
            best_size = r_size
            best_mask = R
        
        if P == 0 and X == 0:
            return
        
        # Pivot: choose from P|X with max degree in P
        U = P | X
        if U == 0:
            return
        pivot = -1
        max_deg = -1
        temp = U
        while temp:
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            deg = popcount(P & adj[v])
            if deg > max_deg:
                max_deg = deg
                pivot = v
            temp ^= lb
        
        # Candidates: P vertices not adjacent to pivot
        cand = P & ~adj[pivot]
        
        while cand:
            if time.time() - start_time > DEADLINE:
                return
            lb = cand & (-cand)
            v = lb.bit_length() - 1
            bron_kerbosch(R | lb, P & adj[v], X & adj[v])
            P ^= lb
            X |= lb
            cand ^= lb

    # Start: R={255}, P=neighbors of 255, X=empty
    # This restricts the search to cliques containing 255, which is required.
    # This is much more efficient than searching the whole graph.
    R = (1 << idx_255)
    P = adj[idx_255]
    X = 0
    
    bron_kerbosch(R, P, X)
    
    result = []
    m = best_mask
    while m:
        lb = m & (-m)
        v = lb.bit_length() - 1
        result.append(candidates[v])
        m ^= lb
    result.sort()
    
    if 255 not in result:
        result.append(255)
        result.sort()
    
    return result
