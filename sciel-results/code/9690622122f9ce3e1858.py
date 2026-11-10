"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    MOD = 101
    DENOMS = list(range(1, 11))
    N = len(DENOMS)
    
    # Precompute inverses
    invs = [pow(x, -1, MOD) for x in DENOMS]
    
    # For each residue, track the minimum length and the corresponding subset
    min_len = [N + 1] * MOD       # initialized to "infinity" (N+1)
    best_subset = [None] * MOD
    
    # Enumerate all 2^10 - 1 = 1023 non-empty subsets via bitmask
    # For each, compute the sum of inverses mod 101
    # Keep the shortest subset for each residue
    for mask in range(1, 1 << N):
        # Count bits and compute sum
        length = 0
        s = 0
        m = mask
        i = 0
        while m:
            if m & 1:
                length += 1
                s += invs[i]
            i += 1
            m >>= 1
        
        s %= MOD
        
        if length < min_len[s]:
            min_len[s] = length
            # Reconstruct the subset
            subset = []
            for j in range(N):
                if mask & (1 << j):
                    subset.append(DENOMS[j])
            best_subset[s] = subset
    
    # Build the result: for each residue, use the shortest subset found
    result = []
    for r in range(MOD):
        if best_subset[r] is not None:
            result.append(list(best_subset[r]))
        else:
            # Fallback: should not happen, but provide a valid non-empty list
            # Use all denominators as a last resort
            result.append(list(DENOMS))
    
    return result
# EVOLVE-BLOCK-END
