"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = start_time + 80.0
    
    n = 256
    values = list(range(1, 257))
    
    # Build adjacency as bitmasks
    adj = [0] * n
    for i in range(n):
        a = values[i]
        for j in range(i + 1, n):
            b = values[j]
            if (a * b) % (a + b) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    def popcount(x):
        return bin(x).count('1')
    
    all_mask = (1 << n) - 1
    
    # Greedy MIS: pick min-degree vertex repeatedly
    def greedy_mis(cands):
        sel = 0
        rem = cands
        while rem:
            best_v = -1
            best_d = n + 1
            m = rem
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                d = popcount(adj[v] & rem)
                if d < best_d:
                    best_d = d
                    best_v = v
                    if d == 0:
                        break
                m ^= lsb
            if best_v < 0:
                break
            sel |= (1 << best_v)
            rem &= ~(1 << best_v)
            rem &= ~adj[best_v]
        return sel
    
    # Get initial solution
    sol = greedy_mis(all_mask)
    best_size = popcount(sol)
    best_mask = sol
    
    # Branch and bound
    def bb(cands, cur_size, cur_mask):
        nonlocal best_size, best_mask
        if time.time() > DEADLINE:
            return
        if cur_size + popcount(cands) <= best_size:
            return
        if cands == 0:
            if cur_size > best_size:
                best_size = cur_size
                best_mask = cur_mask
            return
        # Pick max-degree vertex in cands
        m = cands
        bv = -1
        bd = -1
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            d = popcount(adj[v] & cands)
            if d > bd:
                bd = d
                bv = v
            m ^= lsb
        if bv < 0:
            return
        # Include bv
        nc = cands & ~(1 << bv) & ~adj[bv]
        bb(nc, cur_size + 1, cur_mask | (1 << bv))
        if time.time() > DEADLINE:
            return
        # Exclude bv
        nc = cands & ~(1 << bv)
        bb(nc, cur_size, cur_mask)
    
    bb(all_mask, 0, 0)
    
    result = [values[i] for i in range(n) if (best_mask >> i) & 1]
    if not result:
        result = list(range(1, 257, 2))
    return result

# EVOLVE-BLOCK-END
