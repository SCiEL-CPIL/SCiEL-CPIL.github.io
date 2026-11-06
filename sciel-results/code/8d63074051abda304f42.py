    # Helper for bit counting
    try:
        (1).bit_count()
        def popcount(x):
            return x.bit_count()
    except AttributeError:
        def popcount(x):
            return bin(x).count('1')

    def get_set_bits(mask):
        bits = []
        while mask:
            lsb = mask & -mask
            bits.append(lsb.bit_length() - 1)
            mask ^= lsb
        return bits

    def pivot_greedy():
        P = 0
        for i in range(1, N + 1):
            P |= (1 << i)
        
        clique = []
        mask = 0
        size = 0
        
        while P and elapsed_ok():
            bits = get_set_bits(P)
            if not bits:
                break
            u = random.choice(bits)
            
            Q = P & ~compat[u]
            
            if Q == 0:
                candidates = bits
            else:
                candidates = get_set_bits(Q)
                if not candidates:
                    break
            
            # Heuristic: pick v with max degree in P
            best_v = candidates[0]
            best_deg = -1
            for v in candidates:
                deg = popcount(compat[v] & P)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            v = best_v
            clique.append(v)
            mask |= (1 << v)
            size += 1
            P = P & compat[v]
            
        return clique, mask, size

    best_clique = []
    best_mask = 0
    best_size = 0

    # Run multiple random restarts of pivot_greedy
    num_restarts = 100
    for _ in range(num_restarts):
        if not elapsed_ok():
            break
        clique, mask, size = pivot_greedy()
        
        # Refine with greedy fill
        mask = greedy_fill(clique, mask)
        size = popcount(mask)
        
        if size > best_size:
            best_size = size
            best_clique = list(clique)
            best_mask = mask
