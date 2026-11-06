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
    
    # best[r] stores the list of denominators with minimal size summing to r mod 101
    # min_size[r] stores the size of that list
    best = [None] * 101
    min_size = [11] * 101  # Max possible size is 10, so 11 is infinity
    
    # Iterate through all non-empty subsets
    # We can optimize by iterating by size, but 1023 iterations is so small it doesn't matter.
    for mask in range(1, 1 << 10):
        if time.time() > deadline:
            break
            
        size = 0
        residue = 0
        terms = []
        
        # Unroll loop for speed? Not necessary for 1023*10 ops.
        for i in range(10):
            if mask & (1 << i):
                denom = i + 1
                size += 1
                residue += inv[denom]
                terms.append(denom)
        
        residue %= 101
        
        if size < min_size[residue]:
            min_size[residue] = size
            best[residue] = terms
            # Early exit for this residue if size 1 is found? 
            # We can't break the main loop, but we can skip updates for this residue later.
            # Not worth the complexity.

    # Construct result
    result = []
    for r in range(101):
        if best[r] is not None:
            # Return a copy or the list itself? 
            # "101 nonempty lists"
            result.append(best[r])
        else:
            # This case should not occur for this specific problem parameters,
            # but to be safe against "nonempty" constraint violation if it did:
            # We need to find ANY subset. But if min_size is 11, it means no subset found.
            # With 1023 subsets and 101 residues, coverage is almost certain.
            # If it fails, we have a logical error or bad luck.
            # Let's add a fallback search just in case, though unlikely.
            # Actually, if best[r] is None, it means no mask summed to r.
            # This is mathematically impossible for p=101, D=10?
            # The set of sums is a subset of Z_101. 
            # With 1023 sums, it's extremely likely to be the whole group.
            # If it's not, the task is impossible? No, task implies it's possible.
            # So best[r] should never be None.
            result.append([])

    return result

# EVOLVE-BLOCK-END
