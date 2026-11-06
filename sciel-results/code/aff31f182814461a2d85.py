"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 55.0

    # Precompute inverses of 1..10 mod 101
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
    
    # Precompute all subsets: (residue, size, terms)
    subsets = []
    for mask in range(1, 1 << 10):
        size = 0
        residue = 0
        terms = []
        for i in range(10):
            if mask & (1 << i):
                denom = i + 1
                size += 1
                residue += inv[denom]
                terms.append(denom)
        subsets.append((residue % 101, size, terms))
        
    # best[r] stores the list of denominators with minimal size summing to r mod 101
    # min_size[r] stores the size of that list
    best = [None] * 101
    min_size = [11] * 101  # Max possible size is 10, so 11 is infinity
    
    # Update best for each residue
    for residue, size, terms in subsets:
        if size < min_size[residue]:
            min_size[residue] = size
            best[residue] = terms

    # Construct result
    result = []
    for r in range(101):
        if best[r] is not None:
            result.append(best[r])
        else:
            # Fallback: should not happen
            result.append([])

    return result

# EVOLVE-BLOCK-END
