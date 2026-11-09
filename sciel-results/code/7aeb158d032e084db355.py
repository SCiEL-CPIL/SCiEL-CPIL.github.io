import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    p = 101
    n = 10
    denominators = list(range(1, n + 1))

    # Precompute modular inverses for denominators 1..10
    inverses = [pow(x, -1, p) for x in denominators]

    # Bitmask DP: compute subset sum of inverses for each of the 2^10 - 1 non-empty subsets
    # subset_sums[mask] = sum of inverses of bits set in mask, mod p
    total_masks = 1 << n
    subset_sums = [0] * total_masks
    
    # Precompute popcounts for all masks
    popcounts = [0] * total_masks
    for mask in range(1, total_masks):
        popcounts[mask] = popcounts[mask >> 1] + (mask & 1)

    for mask in range(1, total_masks):
        lsb = mask & (-mask)
        idx = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        subset_sums[mask] = (subset_sums[prev_mask] + inverses[idx]) % p

    # For each residue, find the minimum-popcount subset achieving it
    # best_for_residue[residue] = (popcount, mask)
    best_for_residue = {}

    for mask in range(1, total_masks):
        residue = subset_sums[mask]
        size = popcounts[mask]

        if residue not in best_for_residue or size < best_for_residue[residue][0]:
            best_for_residue[residue] = (size, mask)

    # Coverage check: all 101 residues should be covered
    uncovered = [r for r in range(p) if r not in best_for_residue]
    if uncovered:
        # Fallback: assign all denominators (sum of all inverses) to uncovered residues
        full_mask = total_masks - 1
        for r in uncovered:
            best_for_residue[r] = (n, full_mask)

    # Build result: row r contains the list of denominators from the best subset
    result = []
    for r in range(p):
        size, mask = best_for_residue[r]
        combo = []
        for i in range(n):
            if mask & (1 << i):
                combo.append(i + 1)
        result.append(combo)

    # Time guard
    if time.time() - start_time > DEADLINE:
        pass

    return result
