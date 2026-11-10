"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 256
    # Build conflict graph using bitsets (0-indexed: vertex i = number i+1)
    neighbors = [0] * n
    for a in range(n):
        for b in range(a + 1, n):
            x, y = a + 1, b + 1
            if (x * y) % (x + y) == 0:
                neighbors[a] |= (1 << b)
                neighbors[b] |= (1 << a)
    
    # Helper to count bits
    def popcount(x):
        return bin(x).count('1')
    
    # Helper to compute greedy matching size in subgraph induced by cand
    def greedy_matching_size(cand):
        matched = 0
        remaining = cand
        while remaining:
            lb = remaining & (-remaining)
            v = lb.bit_length() - 1
            # Find a neighbor in remaining
            neigh = neighbors[v] & remaining
            if neigh:
                lb2 = neigh & (-neigh)
                u = lb2.bit_length() - 1
                matched += 1
                remaining &= ~(1 << v)
                remaining &= ~(1 << u)
            else:
                remaining &= ~(1 << v)
        return matched

    # Initial greedy solution to set a good lower bound
    best_mask = 0
    best_size = 0
    remaining = (1 << n) - 1
    while remaining:
        # Pick vertex with min degree in remaining to maximize future choices? 
        # Or just pick first available for speed in init.
        # Let's use a simple greedy: pick vertex, add to set, remove it and neighbors.
        r = remaining
        best_v = -1
        best_d = float('inf')
        while r:
            lb = r & (-r)
            v = lb.bit_length() - 1
            d = popcount(neighbors[v] & remaining)
            if d < best_d:
                best_d = d
                best_v = v
            r ^= lb
        
        best_mask |= (1 << best_v)
        remaining &= ~(1 << best_v)
        remaining &= ~neighbors[best_v]
        
    best_size = popcount(best_mask)
    
    start_time = time.time()
    TIMEOUT = 55
    
    def bnb(candidates, count, current_mask):
        nonlocal best_size, best_mask, start_time
        if time.time() - start_time > TIMEOUT:
            return
        
        if candidates == 0:
            if count > best_size:
                best_size = count
                best_mask = current_mask
            return
            
        # Upper bound: count + (n_cand - max_matching)
        # Max matching >= greedy matching size
        n_cand = popcount(candidates)
        m_size = greedy_matching_size(candidates)
        ub = count + n_cand - m_size
        
        if ub <= best_size:
            return
            
        # Pick vertex with max degree in candidates
        r = candidates
        best_v = -1
        best_d = -1
        while r:
            lb = r & (-r)
            v = lb.bit_length() - 1
            d = popcount(neighbors[v] & candidates)
            if d > best_d:
                best_d = d
                best_v = v
            r ^= lb
            
        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        # New candidates: remove best_v and its neighbors
        new_cand_inc = candidates & ~v_bit & ~neighbors[best_v]
        bnb(new_cand_inc, count + 1, current_mask | v_bit)
        
        # Branch 2: Exclude best_v
        new_cand_exc = candidates & ~v_bit
        bnb(new_cand_exc, count, current_mask)

    all_cand = (1 << n) - 1
    bnb(all_cand, 0, 0)
    
    # Reconstruct result from best_mask
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
# EVOLVE-BLOCK-END
