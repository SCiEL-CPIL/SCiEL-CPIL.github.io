import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s
    
    # 255 = 3 * 5 * 17
    # Any element in the set must be adjacent to 255, i.e., gcd(x, 255) > 1
    # So x must be divisible by 3, 5, or 17
    
    # Build candidate list: all numbers in [1,255] with gcd(x,255)>1
    candidates = []
    for x in range(1, 256):
        if math.gcd(x, 255) > 1:
            candidates.append(x)
    
    # 255 must be in the set
    # Find index of 255
    idx_255 = candidates.index(255)
    
    n = len(candidates)
    
    # Build adjacency bitmasks
    # adj[i] is an integer where bit j is 1 if candidates[i] and candidates[j] are adjacent (gcd > 1)
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    best_size = 0
    best_clique_mask = 0
    
    # Helper to get bit count
    def popcount(x):
        return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')
    
    # Helper to get list of indices from mask
    def mask_to_indices(mask):
        indices = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            indices.append(idx)
            m ^= lsb
        return indices

    # Bron-Kerbosch with pivot
    # clique_mask: current clique
    # cand_mask: candidates to add
    def bron_kerbosch(clique_mask, cand_mask):
        nonlocal best_size, best_clique_mask
        
        if time.time() > DEADLINE:
            return
        
        # Upper bound check
        if popcount(clique_mask) + popcount(cand_mask) <= best_size:
            return
        
        if cand_mask == 0:
            size = popcount(clique_mask)
            if size > best_size:
                best_size = size
                best_clique_mask = clique_mask
            return
        
        # Choose pivot: vertex in cand_mask with max neighbors in cand_mask
        pivot = -1
        max_deg = -1
        temp = cand_mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = popcount(adj[v] & cand_mask)
            if deg > max_deg:
                max_deg = deg
                pivot = v
            temp ^= lsb
        
        # Branch on vertices in cand_mask that are NOT neighbors of pivot
        # This is a standard optimization: any maximal clique must contain
        # a vertex from cand_mask \ N(pivot)
        branch_mask = cand_mask & ~(adj[pivot] if pivot >= 0 else 0)
        
        temp = branch_mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            
            new_clique_mask = clique_mask | lsb
            new_cand_mask = cand_mask & adj[v]
            
            bron_kerbosch(new_clique_mask, new_cand_mask)
            
            cand_mask ^= lsb # Remove v from candidates for subsequent branches
            
            if time.time() > DEADLINE:
                return
            
            temp &= temp - 1 # Remove lsb from temp
        
    # Initialize
    # We MUST include 255. 
    # So start with clique = {255}, candidates = neighbors of 255.
    init_clique_mask = 1 << idx_255
    init_cand_mask = adj[idx_255]
    
    bron_kerbosch(init_clique_mask, init_cand_mask)
    
    # Convert best_clique_mask to list of values
    result_indices = mask_to_indices(best_clique_mask)
    result_values = [candidates[i] for i in result_indices]
    result_values.sort()
    
    # Verify
    if 255 not in result_values:
        # Should not happen if we initialized correctly, but just in case
        result_values.append(255)
        
    return result_values
