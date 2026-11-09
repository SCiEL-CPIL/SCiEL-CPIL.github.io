"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

def construct():
    start_time = time.time()
    LIMIT = 85.0  # Safety margin before 90s kill

    p = 101
    denominators = list(range(1, 11))
    n = 10

    # Precompute modular inverses
    inverses = [pow(x, -1, p) for x in denominators]

    # Precompute sum of inverses for each subset (represented as bitmask)
    # There are 2^10 = 1024 subsets total
    subset_sums = [0] * (1 << n)
    subset_masks = [0] * (1 << n)
    
    for mask in range(1, 1 << n):
        # Use lowest set bit to compute incrementally
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        subset_sums[mask] = (subset_sums[prev_mask] + inverses[idx]) % p
        subset_masks[mask] = mask

    # For each residue, find the subset with minimum size (popcount)
    # best_for_residue[residue] = (popcount, mask)
    best_for_residue = {}  # residue -> (size, mask)
    
    for mask in range(1, 1 << n):
        residue = subset_sums[mask]
        size = bin(mask).count('1')
        
        if residue not in best_for_residue or size < best_for_residue[residue][0]:
            best_for_residue[residue] = (size, mask)

    # Check coverage
    uncovered = [r for r in range(p) if r not in best_for_residue]
    
    # If some residues are uncovered, use fallback
    if uncovered:
        # For uncovered residues, try to find any subset (even larger)
        # Since we've checked all subsets, if still uncovered, use all denominators
        for r in uncovered:
            best_for_residue[r] = (n, (1 << n) - 1)

    # Construct the result
    result = []
    for r in range(p):
        if r in best_for_residue:
            size, mask = best_for_residue[r]
            # Convert mask to list of denominators
            combo = []
            for i in range(n):
                if mask & (1 << i):
                    combo.append(i + 1)
            result.append(combo)
        else:
            # Fallback: use all denominators
            result.append(list(denominators))
    
    return result
# EVOLVE-BLOCK-END
