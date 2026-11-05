import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 255
    # Candidates: numbers in [1,255] with gcd(v,255) > 1
    candidates = [v for v in range(1, n+1) if math.gcd(v, 255) > 1]
    cand_set = set(candidates)
    
    # Precompute gcd compatibility between all candidate pairs
    # compat[a][b] = True if gcd(a,b) > 1
    # Use a dict or 2D array indexed by candidate position
    cand_idx = {v: i for i, v in enumerate(candidates)}
    m = len(candidates)
    
    # Build adjacency as bitsets for speed
    # adj[i] = bitmask of which candidates j have gcd(candidates[i], candidates[j]) > 1
    adj = [0] * m
    for i in range(m):
        for j in range(i+1, m):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Start with 255
    idx_255 = cand_idx[255]
    clique_mask = 1 << idx_255
    clique_list = [255]
    
    # Available candidates: those compatible with all in clique
    # Initially: those with gcd(v, 255) > 1, which is all candidates
    available = adj[idx_255]  # bitmask of candidates compatible with 255
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Find best vertex in available to add
        best_i = -1
        best_deg = -1
        # Iterate through available bits
        avail = available
        while avail:
            # Get lowest set bit
            lb = avail & (-avail)
            i = lb.bit_length() - 1
            avail ^= lb
            
            # Degree of i within available
            deg = (adj[i] & available).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_i = i
        
        if best_i == -1:
            break
        
        clique_mask |= (1 << best_i)
        clique_list.append(candidates[best_i])
        # Update available: intersect with adj[best_i]
        available &= adj[best_i]
    
    return clique_list
