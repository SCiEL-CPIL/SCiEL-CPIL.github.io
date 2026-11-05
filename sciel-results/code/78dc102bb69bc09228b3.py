<<<<<<< SEARCH
    ds = get_direct_sum()
    
    # Initialize with Greedy
    uncovered = np.ones(total, dtype=np.int8)
    code = []
    while uncovered.sum() > 0:
        if time.time() > deadline:
            break
        counts = np.sum(uncovered[nbr], axis=1)
        best = int(np.argmax(counts))
        if counts[best] == 0:
            break
        code.append(best)
        uncovered[nbr[best]] = 0
=======
    ds = get_direct_sum()
    
    # Compute a 2-packing (maximal set of words at Hamming distance >= 3)
    # using greedy with randomization as structural guide
    def compute_2packing(seed_val=42):
        rng_p = np.random.default_rng(seed_val)
        order = rng_p.permutation(total)
        packing = []
        in_packing = np.zeros(total, dtype=np.int8)
        for idx in order:
            # Check if idx is at distance >= 3 from all in packing
            ok = True
            for p in packing:
                # Compute Hamming distance
                d = 0
                for pos in range(6):
                    if (idx // powers[pos]) % 3 != (p // powers[pos]) % 3:
                        d += 1
                        if d > 2:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                packing.append(idx)
                in_packing[idx] = 1
        return packing

    # Compute multiple 2-packings for diversity
    packings = [compute_2packing(s) for s in range(3)]
    
    # Initialize with Greedy guided by 2-packing structure
    # First add packing elements, then fill gaps greedily
    uncovered = np.ones(total, dtype=np.int8)
    code = []
    
    # Add packing elements first (they're structurally well-distributed)
    for p in packings[0]:
        uncovered[nbr[p]] = 0
        code.append(p)
    
    # Fill remaining gaps greedily
    while uncovered.sum() > 0:
        if time.time() > deadline:
            break
        counts = np.sum(uncovered[nbr], axis=1)
        # Only consider words not already in code
        in_code_mask = np.zeros(total, dtype=np.int8)
        for c in code:
            in_code_mask[c] = 1
        counts[in_code_mask == 1] = -1
        best = int(np.argmax(counts))
        if counts[best] <= 0:
            break
        code.append(best)
        uncovered[nbr[best]] = 0
>>>>>>> REPLACE

