import time

def construct():
    """
    Returns a list of 101 lists. Each list contains distinct denominators in 1..10
    such that the sum of their modular inverses modulo 101 equals the index.
    Minimizes the largest row length.
    """
    MOD = 101
    MAX_DENOM = 10
    denominators = list(range(1, MAX_DENOM + 1))
    
    # Precompute modular inverses
    inv = [0] * (MAX_DENOM + 1)
    for d in denominators:
        inv[d] = pow(d, -1, MOD)
    
    n = MAX_DENOM
    total_masks = 1 << n
    subset_sum = [0] * total_masks
    subset_len = [0] * total_masks
    
    for mask in range(1, total_masks):
        lb = mask & (-mask)
        bit_idx = lb.bit_length() - 1
        prev_mask = mask ^ lb
        subset_sum[mask] = (subset_sum[prev_mask] + inv[bit_idx + 1]) % MOD
        subset_len[mask] = subset_len[prev_mask] + 1
    
    # For each residue, find the shortest subset that sums to it
    # best[r] = mask of shortest subset summing to r, or -1 if not found
    best = [-1] * MOD
    
    # Iterate by length to find shortest first
    for length in range(1, n + 1):
        for mask in range(1, total_masks):
            if subset_len[mask] != length:
                continue
            r = subset_sum[mask]
            if best[r] == -1:
                best[r] = mask
        
        # Check if all residues are covered
        if all(b != -1 for b in best):
            break
    
    # Build results by decoding masks
    results = []
    for r in range(MOD):
        mask = best[r]
        if mask != -1:
            elems = []
            for bit in range(n):
                if mask & (1 << bit):
                    elems.append(bit + 1)
            results.append(elems)
        else:
            # Fallback: find any subset
            for mask in range(1, total_masks):
                if subset_sum[mask] == r:
                    elems = []
                    for bit in range(n):
                        if mask & (1 << bit):
                            elems.append(bit + 1)
                    results.append(elems)
                    break
            else:
                results.append([1])
    
    return results
