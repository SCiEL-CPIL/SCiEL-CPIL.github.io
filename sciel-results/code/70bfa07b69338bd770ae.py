"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    
    # Precompute modular inverses of 1..10 mod 101
    inv = [pow(x, -1, 101) for x in range(1, 11)]
    
    # Precompute all non-empty subsets: (size, residue, mask)
    # Use efficient bit manipulation
    all_subsets = []
    for mask in range(1, 1 << 10):
        size = bin(mask).count('1')
        residue = 0
        m = mask
        x = 1
        while m:
            if m & 1:
                residue += inv[x - 1]
            x += 1
            m >>= 1
        residue %= 101
        all_subsets.append((size, residue, mask))
    
    # Group subsets by residue, keeping only the minimum size for each residue
    # For each residue, we only need the smallest subset
    best_for_residue = {}  # residue -> (min_size, mask)
    for size, residue, mask in all_subsets:
        if residue not in best_for_residue or size < best_for_residue[residue][0]:
            best_for_residue[residue] = (size, mask)
    
    # For each residue, find the minimum subset size needed
    min_size_for_residue = {}
    for r in range(101):
        if r in best_for_residue:
            min_size_for_residue[r] = best_for_residue[r][0]
        else:
            min_size_for_residue[r] = 10  # fallback
    
    # The global maximum of these minima is our target max row length
    target_max_len = max(min_size_for_residue.values())
    
    # Build result: for each residue, pick the minimum-size subset
    result = []
    for r in range(101):
        if r in best_for_residue:
            size, mask = best_for_residue[r]
            terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
            result.append(terms)
        else:
            # Fallback: use all denominators
            result.append(list(range(1, 11)))
    
    return result
# EVOLVE-BLOCK-END
