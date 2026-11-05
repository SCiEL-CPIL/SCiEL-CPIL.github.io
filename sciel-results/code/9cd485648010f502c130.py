import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Modular inverses of 1..10 mod 101
    # i^{-1} mod 101
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, 100, 101)  # Fermat's little theorem: i^(p-2) = i^-1 mod p

    # We want to find, for each residue r in 0..100, a non-empty subset of {1..10}
    # such that the sum of inverses is r mod 101.
    # We want to minimize the maximum size of these subsets.
    
    # Since the space of subsets is small (2^10 - 1 = 1023), we can precompute
    # the sum and size for every subset.
    
    # min_size[r] will store the minimum size of a subset summing to r.
    min_size = [11] * 101
    # best_mask[r] stores the mask of the subset achieving min_size[r].
    best_mask = [0] * 101

    # Iterate over all non-empty subsets
    for mask in range(1, 1 << 10):
        # Calculate sum and popcount for this mask
        s = 0
        count = 0
        m = mask
        # Unroll loop for speed, or just iterate bits
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s = (s + inv[idx + 1]) % 101
            count += 1
            m ^= lsb
            
        # Update best for residue s
        if count < min_size[s]:
            min_size[s] = count
            best_mask[s] = mask

    # Construct the result
    result = [None] * 101
    for r in range(101):
        if min_size[r] == 11:
            # Fallback, though shouldn't happen
            result[r] = [1] 
        else:
            mask = best_mask[r]
            terms = []
            for i in range(1, 11):
                if mask & (1 << (i - 1)):
                    terms.append(i)
            result[r] = terms
            
    return result
