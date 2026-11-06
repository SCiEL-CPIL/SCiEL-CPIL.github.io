def _find_violation(adj_bits, n, rng):
    # 1. Random probe
    # Check random subsets of sizes 6 to min(n, 12)
    max_probe_size = min(n, 12)
    for _ in range(50):
        size = rng.randint(6, max_probe_size)
        if size > n: continue
        subset = set(rng.sample(range(n), size))
        mask = 0
        for v in subset:
            mask |= (1 << v)
        
        # Check regularity
        degs = []
        for v in subset:
            degs.append((adj_bits[v] & mask).bit_count())
        if len(set(degs)) == 1:
            return mask

    # 2. Full scan (if probe failed)
    for size in range(6, n + 1):
        for combo in itertools.combinations(range(n), size):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            
            degs = []
            for v in combo:
                degs.append((adj_bits[v] & mask).bit_count())
            if len(set(degs)) == 1:
                return mask
    return None
