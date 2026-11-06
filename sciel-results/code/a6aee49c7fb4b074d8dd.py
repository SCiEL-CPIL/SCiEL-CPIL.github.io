import itertools
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
    
    # Precompute sum for each subset using bitmask
    # subset_sum[mask] = sum of inverses of bits set in mask, mod 101
    # subset_len[mask] = number of bits set in mask
    n = MAX_DENOM
    total_masks = 1 << n
    subset_sum = [0] * total_masks
    subset_len = [0] * total_masks
    subset_elems = [None] * total_masks
    
    for mask in range(1, total_masks):
        # Find lowest set bit
        lb = mask & (-mask)
        bit_idx = lb.bit_length() - 1  # 0-indexed
        prev_mask = mask ^ lb
        subset_sum[mask] = (subset_sum[prev_mask] + inv[bit_idx + 1]) % MOD
        subset_len[mask] = subset_len[prev_mask] + 1
        subset_elems[mask] = subset_elems[prev_mask] + [bit_idx + 1] if subset_elems[prev_mask] is not None else [bit_idx + 1]
    
    # For each residue, find the shortest subset that sums to it
    best = [None] * MOD  # best[r] = (length, list_of_denoms)
    
    # Iterate by length to find shortest first
    for length in range(1, n + 1):
        for mask in range(1, total_masks):
            if subset_len[mask] != length:
                continue
            r = subset_sum[mask]
            if best[r] is None:
                best[r] = (length, subset_elems[mask])
        
        # Check if all residues are covered
        if all(b is not None for b in best):
            break
    
    # Build results
    results = []
    for r in range(MOD):
        if best[r] is not None:
            results.append(best[r][1])
        else:
            # Fallback: should not happen, but find any subset
            found = False
            for mask in range(1, total_masks):
                if subset_sum[mask] == r:
                    results.append(subset_elems[mask])
                    found = True
                    break
            if not found:
                results.append([1])
    
    return results
