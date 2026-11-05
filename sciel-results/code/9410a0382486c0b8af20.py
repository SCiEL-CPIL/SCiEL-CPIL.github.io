import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe
    
    N = 256
    # Map value v to index i = v - 1
    
    # 1. Identify bad vertices (2*v is a perfect square)
    bad_mask = 0
    good_indices = []
    for v in range(1, N + 1):
        if int(math.isqrt(2 * v))**2 == 2 * v:
            # Bad vertex, cannot be in set
            pass
        else:
            good_indices.append(v - 1)
            
    M = len(good_indices)
    if M == 0:
        return []
        
    # Map original index to compact index? 
    # It's easier to work with the original 0..255 indices and just mask out bad ones.
    # Let's stick to 0..255 indices.
    
    # Initial mask of all good vertices
    current_mask = 0
    for idx in good_indices:
        current_mask |= (1 << idx)
        
    # 2. Build adjacency masks
    # adj[i] is a bitmask of j such that i and j are connected (sum is square)
    # We only care about connections between good vertices.
    adj = [0] * N
    for i in range(N):
        if not (current_mask & (1 << i)):
            continue
        val_i = i + 1
        for j in range(i + 1, N):
            if not (current_mask & (1 << j)):
                continue
            val_j = j + 1
            s = val_i + val_j
            r = int(math.isqrt(s))
            if r * r == s:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    best_size = 0
    best_mask = 0
    # Heuristic: Start with a greedy solution to get a good lower bound
    # Greedy: Pick vertex with min degree, add to set, remove neighbors.
    # Or just start with 0, B&B will find a solution quickly.
    # Let's try a quick greedy to set best_size.
    temp_mask = current_mask
    greedy_set = []
    while temp_mask:
        # Find vertex with min degree in temp_mask
        min_deg = float('inf')
        min_v = -1
        mm = temp_mask
        while mm:
            v = (mm & -mm).bit_length() - 1
            mm &= mm - 1
            deg = (adj[v] & temp_mask).bit_count()
            if deg < min_deg:
                min_deg = deg
                min_v = v
        if min_v != -1:
            greedy_set.append(min_v)
            temp_mask &= ~adj[min_v]
            temp_mask &= ~(1 << min_v)
    if greedy_set:
        best_size = len(greedy_set)
        gm = 0
        for v in greedy_set:
            gm |= (1 << v)
        best_mask = gm

    # B&B
    def popcount(x):
        return x.bit_count()

    def solve(mask, current_size, current_set_mask):
        nonlocal best_size, best_mask
        if time.time() > deadline:
            return
        
        if mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_set_mask
            return
            
        # Prune
        if current_size + popcount(mask) <= best_size:
            return
            
        # Select vertex v from mask with highest degree in mask
        max_deg = -1
        v = -1
        mm = mask
        while mm:
            vv = (mm & -mm).bit_length() - 1
            mm &= mm - 1
            deg = (adj[vv] & mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = vv
                
        # Branch 1: Include v
        # New mask: remove v and its neighbors
        new_mask_inc = mask & ~adj[v] & ~(1 << v)
        solve(new_mask_inc, current_size + 1, current_set_mask | (1 << v))
        
        # Branch 2: Exclude v
        new_mask_exc = mask & ~(1 << v)
        solve(new_mask_exc, current_size, current_set_mask)

    solve(current_mask, 0, 0)
    
    # Reconstruct result from best_mask
    result = []
    bm = best_mask
    while bm:
        v = (bm & -bm).bit_length() - 1
        bm &= bm - 1
        result.append(v + 1)
        
    return sorted(result)
