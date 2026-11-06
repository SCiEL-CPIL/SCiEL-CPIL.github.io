"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Precompute modular inverses for denominators 1 to 10 modulo 101
    inv = [0] * 11
    for x in range(1, 11):
        inv[x] = pow(x, -1, 101)
    
    # Enumerate all non-empty subsets of {1,...,10}
    # For each subset, compute the sum of inverses mod 101
    # We want to find, for each residue r in 0..100, a subset that sums to r
    # and minimize the maximum subset size across all residues.
    
    # Strategy:
    # 1. First, find all subsets and their residues.
    # 2. For each residue, track the minimum-size subset that achieves it.
    # 3. The answer is the maximum of these minimum sizes.
    
    # Since there are only 2^10 - 1 = 1023 non-empty subsets, we can enumerate all.
    # For each residue, we want the smallest subset (by size) that sums to it.
    
    # min_size[r] = minimum size of a subset summing to r mod 101
    min_size = [100] * 101  # initialize with large value
    best_subset = [None] * 101
    
    # Enumerate subsets by size, from smallest to largest
    # This way, the first time we hit a residue, it's with the smallest possible size
    found = [False] * 101
    
    for size in range(1, 11):
        # Generate all combinations of size 'size' from {1,...,10}
        for combo in itertools.combinations(range(1, 11), size):
            s = sum(inv[x] for x in combo) % 101
            if not found[s]:
                found[s] = True
                min_size[s] = size
                best_subset[s] = list(combo)
        
        # Check if all residues are found
        if all(found):
            break
        
        # Check deadline
        if time.time() > DEADLINE:
            break
    
    # Verify all residues are covered
    # If some residues are not covered (shouldn't happen with 1023 subsets and 101 residues),
    # we need to handle that. But with 1023 subsets and 101 residues, by pigeonhole,
    # some residues will have multiple subsets. The question is whether every residue is hit.
    
    # Actually, let's verify: do all 101 residues appear among the 1023 subset sums?
    # If not, we have a problem. Let's check.
    
    if not all(found):
        # Fallback: use the original approach but with minimum size
        # Re-enumerate all subsets and for each residue, keep the smallest subset
        for mask in range(1, 1 << 10):
            terms = [x for x in range(1, 11) if (mask >> (x-1)) & 1]
            residue = sum(inv[x] for x in terms) % 101
            if len(terms) < min_size[residue]:
                min_size[residue] = len(terms)
                best_subset[residue] = terms
        
        # If still some residues uncovered, use any subset (shouldn't happen)
        for r in range(101):
            if best_subset[r] is None:
                # Find any subset that sums to r
                for mask in range(1, 1 << 10):
                    terms = [x for x in range(1, 11) if (mask >> (x-1)) & 1]
                    residue = sum(inv[x] for x in terms) % 101
                    if residue == r:
                        best_subset[r] = terms
                        min_size[r] = len(terms)
                        break
    
    # Build the result: row r contains best_subset[r]
    result = [best_subset[r] for r in range(101)]
    
    return result
# EVOLVE-BLOCK-END
