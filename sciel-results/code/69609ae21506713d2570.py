import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Numbers in [1,255] that share a factor with 255=3*5*17
    candidates = [i for i in range(1, 256) if math.gcd(i, 255) > 1]
    if 255 in candidates:
        candidates.remove(255)
    n = len(candidates)

    # Build adjacency bitmasks: adj[i] has bit j set if gcd(candidates[i], candidates[j])>1
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    def popcount(x):
        return bin(x).count("1")

    # Initial clique: all multiples of 3 (they all share factor 3)
    init_mask = 0
    for i in range(n):
        if candidates[i] % 3 == 0:
            init_mask |= (1 << i)
    best_size = popcount(init_mask)
    best_mask = init_mask

    # Greedy: iteratively add vertex with max connections to current clique
    greedy_mask = 0
    remaining = (1 << n) - 1
    while remaining:
        if time.time() > DEADLINE:
            break
        best_v = -1
        best_conn = -1
        tmp = remaining
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            conn = popcount(adj[v] & greedy_mask)
            if conn > best_conn:
                best_conn = conn
                best_v = v
            tmp ^= lsb
        if best_conn < 0:
            break
        greedy_mask |= (1 << best_v)
        remaining &= adj[best_v]

    if popcount(greedy_mask) > best_size:
        best_size = popcount(greedy_mask)
        best_mask = greedy_mask

    # Try to extend the best found clique by finding vertices that connect to all in best_mask
    # and then greedily adding more
    def extend_clique(clique_mask):
        nonlocal best_size, best_mask
        current = clique_mask
        # Find all vertices that connect to all in current
        ext_candidates = 0
        tmp = (1 << n) - 1
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            if (adj[v] & current) == current:
                ext_candidates |= lsb
            tmp ^= lsb
        
        # Greedily add from ext_candidates
        while ext_candidates:
            if time.time() > DEADLINE:
                break
            best_v = -1
            best_conn = -1
            tmp = ext_candidates
            while tmp:
                lsb = tmp & (-tmp)
                v = lsb.bit_length() - 1
                conn = popcount(adj[v] & current)
                if conn > best_conn:
                    best_conn = conn
                    best_v = v
                tmp ^= lsb
            if best_v < 0:
                break
            current |= (1 << best_v)
            # Update ext_candidates to only those that connect to new current
            new_ext = 0
            tmp = ext_candidates
            while tmp:
                lsb = tmp & (-tmp)
                v = lsb.bit_length() - 1
                if (adj[v] & current) == current:
                    new_ext |= lsb
                tmp ^= lsb
            ext_candidates = new_ext
        
        if popcount(current) > best_size:
            best_size = popcount(current)
            best_mask = current
        return current

    # Try extending from the greedy result
    extend_clique(best_mask)

    # Try multiple random restarts of greedy to find better starting cliques
    import random
    random.seed(42)
    
    for restart in range(50):
        if time.time() > DEADLINE:
            break
        
        # Random greedy with different ordering
        greedy_mask_r = 0
        remaining_r = (1 << n) - 1
        
        # Shuffle vertices for diversity
        order = list(range(n))
        random.shuffle(order)
        
        while remaining_r:
            if time.time() > DEADLINE:
                break
            best_v = -1
            best_conn = -1
            # Prefer vertices that appear earlier in our shuffled order among those with max connections
            for v in order:
                if not (remaining_r & (1 << v)):
                    continue
                conn = popcount(adj[v] & greedy_mask_r)
                if conn > best_conn:
                    best_conn = conn
                    best_v = v
            if best_v < 0:
                break
            greedy_mask_r |= (1 << best_v)
            remaining_r &= adj[best_v]
        
        if popcount(greedy_mask_r) > best_size:
            best_size = popcount(greedy_mask_r)
            best_mask = greedy_mask_r
        
        # Try to extend this clique
        extend_clique(best_mask)

    # Convert best_mask back to list, ensure 255 is included
    result = [255]
    tmp = best_mask
    while tmp:
        lsb = tmp & (-tmp)
        idx = lsb.bit_length() - 1
        result.append(candidates[idx])
        tmp ^= lsb

    return result
# EVOLVE-BLOCK-END
