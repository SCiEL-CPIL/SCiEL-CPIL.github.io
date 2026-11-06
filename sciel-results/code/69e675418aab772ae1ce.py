"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0
    
    p = 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, p)
    
    # Precompute all 2^10 subset sums using bitmask enumeration
    # mask bit i (0-indexed) corresponds to denominator (i+1)
    subset_sum = [0] * (1 << 10)
    subset_bits = [0] * (1 << 10)
    
    for mask in range(1, 1 << 10):
        # Use lowest set bit to build incrementally
        lb = mask & (-mask)
        idx = lb.bit_length() - 1  # 0-indexed, so denominator = idx + 1
        prev = mask ^ lb
        subset_sum[mask] = (subset_sum[prev] + inv[idx + 1]) % p
        subset_bits[mask] = mask
    
    # For each residue, find the minimum-size subset that achieves it
    min_len = [11] * p
    best_subsets = [None] * p
    
    # Process masks in order of popcount (size) to ensure we find min size first
    # Group masks by their popcount
    by_size = [[] for _ in range(11)]
    for mask in range(1, 1 << 10):
        sz = bin(mask).count('1')
        by_size[sz].append(mask)
    
    for size in range(1, 11):
        if time.time() > deadline:
            break
        
        # Check if all residues are covered
        if all(ml < 11 for ml in min_len):
            break
        
        for mask in by_size[size]:
            s = subset_sum[mask]
            if size < min_len[s]:
                min_len[s] = size
                # Convert mask to list of denominators
                denoms = []
                for i in range(10):
                    if mask & (1 << i):
                        denoms.append(i + 1)
                best_subsets[s] = denoms
    
    # Fallback for any uncovered residues (shouldn't happen, but safety)
    for r in range(p):
        if best_subsets[r] is None:
            # Find any subset that gives residue r
            for mask in range(1, 1 << 10):
                if subset_sum[mask] == r:
                    denoms = []
                    for i in range(10):
                        if mask & (1 << i):
                            denoms.append(i + 1)
                    best_subsets[r] = denoms
                    min_len[r] = len(denoms)
                    break
    
    return [best_subsets[r] for r in range(p)]
# EVOLVE-BLOCK-END
