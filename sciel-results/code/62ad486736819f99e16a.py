"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 256
    # Build conflict graph using bitsets (0-indexed: vertex i = number i+1)
    conflicts = [0] * n
    for a in range(n):
        for b in range(a + 1, n):
            x, y = a + 1, b + 1
            if (x * y) % (x + y) == 0:
                conflicts[a] |= (1 << b)
                conflicts[b] |= (1 << a)
    
    # Vertices with no conflicts can always be included
    isolated = [i for i in range(n) if conflicts[i] == 0]
    non_isolated = [i for i in range(n) if conflicts[i] != 0]
    
    # Focus B&B on non-isolated vertices
    m = len(non_isolated)
    
    # Build compact conflict graph
    cconf = [0] * m
    for i in range(m):
        for j in range(i + 1, m):
            vi, vj = non_isolated[i], non_isolated[j]
            if (conflicts[vi] >> vj) & 1:
                cconf[i] |= (1 << j)
                cconf[j] |= (1 << i)
    
    # Greedy initial solution on compact graph
    greedy_mask = 0
    remaining = (1 << m) - 1
    while remaining:
        r = remaining
        best_v = -1
        best_d = float('inf')
        while r:
            lb = r & (-r)
            v = lb.bit_length() - 1
            d = bin(cconf[v] & remaining).count('1')
            if d < best_d:
                best_d = d
                best_v = v
            r ^= lb
        greedy_mask |= (1 << best_v)
        remaining &= ~(1 << best_v)
        remaining &= ~cconf[best_v]
    
    best_size = bin(greedy_mask).count('1')
    best_mask = greedy_mask
    start_time = time.time()
    TIMEOUT = 55
    
    def bnb(candidates, count):
        nonlocal best_size, best_mask, start_time
        if time.time() - start_time > TIMEOUT:
            return
        if candidates == 0:
            if count > best_size:
                best_size = count
                best_mask = 0
            return
        if count + bin(candidates).count('1') <= best_size:
            return
        # Pick vertex with max degree in candidates
        r = candidates
        best_v = -1
        best_d = -1
        while r:
            lb = r & (-r)
            v = lb.bit_length() - 1
            d = bin(cconf[v] & candidates).count('1')
            if d > best_d:
                best_d = d
                best_v = v
            r ^= lb
        v_bit = 1 << best_v
        # Branch 1: include
        bnb(candidates & ~v_bit & ~cconf[best_v], count + 1)
        # Branch 2: exclude
        bnb(candidates & ~v_bit, count)
    
    all_cand = (1 << m) - 1
    bnb(all_cand, 0)
    
    # Reconstruct: use greedy to get a valid set, then try to improve to match best_size
    result_set = set(isolated)
    remaining = (1 << m) - 1
    while remaining:
        r = remaining
        best_v = -1
        best_d = float('inf')
        while r:
            lb = r & (-r)
            v = lb.bit_length() - 1
            d = bin(cconf[v] & remaining).count('1')
            if d < best_d:
                best_d = d
                best_v = v
            r ^= lb
        result_set.add(non_isolated[best_v])
        remaining &= ~(1 << best_v)
        remaining &= ~cconf[best_v]
    
    result = sorted([v + 1 for v in result_set])
    return result
# EVOLVE-BLOCK-END
