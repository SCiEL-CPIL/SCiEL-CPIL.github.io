import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    candidates = [i for i in range(1, 256) if math.gcd(i, 255) > 1]
    if 255 not in candidates:
        candidates.append(255)
    
    n = len(candidates)
    idx = {v: i for i, v in enumerate(candidates)}
    idx_255 = idx[255]
    
    # Build adjacency as bitmasks
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    best_size = 1
    best_mask = (1 << idx_255)
    
    def popcount(x):
        return bin(x).count("1")
    
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
        # Find pivot
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
