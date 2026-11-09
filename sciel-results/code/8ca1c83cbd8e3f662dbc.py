import time

def construct():
    """
    Construct 101 nonempty lists of distinct denominators in 1..10 whose
    modular inverses sum to the row index modulo 101.
    
    The problem decomposes into finding, for each residue r, the smallest subset 
    of {1, ..., 10} whose inverse sum is r mod 101. The global objective is to 
    minimize the maximum size among these subsets. Since rows are independent 
    (denominators can recur across rows), the optimal solution is to pick the 
    minimum-size subset for each residue independently.
    
    We enumerate all 2^10 - 1 = 1023 non-empty subsets, compute their inverse 
    sums and sizes, and keep the smallest size for each residue.
    """
    MOD = 101
    N = 10
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # 50s margin for 60s limit
    
    # Pre-compute modular inverses of 1..10 mod 101
    inv = [0] * (N + 1)
    for d in range(1, N + 1):
        inv[d] = pow(d, MOD - 2, MOD)
        
    # min_size[r] will store the minimum size of a subset summing to r
    # best_mask[r] will store the bitmask of that subset
    min_size = [N + 1] * MOD
    best_mask = [0] * MOD
    
    # Enumerate all non-empty subsets
    for mask in range(1, 1 << N):
        # Check deadline
        if time.time() > deadline:
            break
            
        s = 0
        size = 0
        # Efficient sum calculation
        m = mask
        i = 0
        while m:
            if m & 1:
                s = (s + inv[i + 1]) % MOD
                size += 1
            i += 1
            m >>= 1
        
        if size < min_size[s]:
            min_size[s] = size
            best_mask[s] = mask

    # Construct the result lists
    result = []
    for r in range(MOD):
        if best_mask[r] != 0:
            # Reconstruct the list of denominators from the mask
            terms = []
            m = best_mask[r]
            i = 0
            while m:
                if m & 1:
                    terms.append(i + 1)
                i += 1
                m >>= 1
            result.append(terms)
        else:
            # Fallback (should not happen)
            # Find any subset, even if not minimal
            found = False
            for mask in range(1, 1 << N):
                s = 0
                m = mask
                i = 0
                while m:
                    if m & 1:
                        s = (s + inv[i + 1]) % MOD
                    i += 1
                    m >>= 1
                if s == r:
                    terms = []
                    m = mask
                    i = 0
                    while m:
                        if m & 1:
                            terms.append(i + 1)
                        i += 1
                        m >>= 1
                    result.append(terms)
                    found = True
                    break
            if not found:
                result.append([1])
                
    return result
