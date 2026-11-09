import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    p = 101
    # Precompute inverses for denominators 1..10 modulo 101
    # invs[i] is the inverse of (i+1) mod 101
    invs = [pow(x, -1, p) for x in range(1, 11)]
    
    # min_size[r] will store the minimum size of a subset that sums to r
    # min_mask[r] will store the bitmask of that subset
    min_size = [11] * p  # 11 is larger than max possible size 10
    min_mask = [0] * p
    
    # Iterate through all non-empty subsets of {0, ..., 9}
    # There are 1023 such subsets.
    for mask in range(1, 1 << 10):
        if time.time() > deadline:
            break
            
        # Calculate sum and size
        s = 0
        size = 0
        m = mask
        i = 0
        while m:
            if m & 1:
                s = (s + invs[i]) % p
                size += 1
            i += 1
            m >>= 1
        
        # If this subset size is smaller than the current best for this residue, update
        if size < min_size[s]:
            min_size[s] = size
            min_mask[s] = mask
    
    # Build the result
    # For each residue r, we need the list of denominators (1-indexed) from the best mask
    result = []
    for r in range(p):
        mask = min_mask[r]
        # If for some reason no subset was found (should not happen), fallback to [1]
        if mask == 0:
            result.append([1])
            continue
            
        terms = []
        m = mask
        i = 0
        while m:
            if m & 1:
                terms.append(i + 1)
            i += 1
            m >>= 1
        result.append(terms)
        
    return result
