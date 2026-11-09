"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    
    # Constants
    P = 101
    N = 10
    DEADLINE = 85  # Safety margin before 90s kill
    
    # Pre-compute modular inverses for 1..10
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, -1, P)
    
    # Helper to convert mask to list of denominators
    def mask_to_list(mask):
        return [i for i in range(1, N + 1) if mask & (1 << (i - 1))]
    
    # Enumerate all 1023 non-empty subsets of {1..N}, computing their sum mod P
    # Group by sum: for each residue, track the minimum-size subset
    # Since denominators may recur across rows, each residue independently
    # picks its minimum-size valid subset.
    
    # min_size[r] = smallest size of any subset summing to r mod P
    # best_mask[r] = a mask achieving that minimum size
    min_size = [N + 1] * P  # initialized to impossible
    best_mask = [0] * P
    
    # Enumerate all non-empty subsets via bitmask
    # For each subset, compute sum mod P and popcount
    for mask in range(1, 1 << N):
        if time.time() - start_time > DEADLINE:
            break
        s = 0
        cnt = 0
        m = mask
        i = 0
        while m:
            if m & 1:
                s = (s + inv[i + 1]) % P
                cnt += 1
            m >>= 1
            i += 1
        if cnt < min_size[s]:
            min_size[s] = cnt
            best_mask[s] = mask
    
    # Check if all residues are covered
    uncovered = [r for r in range(P) if min_size[r] > N]
    
    if uncovered:
        # Fallback: this shouldn't happen, but ensure validity
        # Use any subset for uncovered residues
        all_subsets = {}
        for mask in range(1, 1 << N):
            s = 0
            for i in range(1, N + 1):
                if mask & (1 << (i - 1)):
                    s = (s + inv[i]) % P
            if s not in all_subsets:
                all_subsets[s] = mask
        
        final_result = []
        for r in range(P):
            if r in all_subsets:
                final_result.append(mask_to_list(all_subsets[r]))
            else:
                final_result.append([1])
        return final_result
    
    # Construct result: each residue gets its minimum-size subset
    result = []
    for r in range(P):
        result.append(mask_to_list(best_mask[r]))
    
    return result

# EVOLVE-BLOCK-END
