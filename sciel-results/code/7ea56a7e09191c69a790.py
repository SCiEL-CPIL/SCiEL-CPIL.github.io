import itertools

def construct():
    # Precompute modular inverses for 1..10
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)
        
    # min_size[r] will store the minimum size of a subset summing to r
    # best_mask[r] will store one such mask
    min_size = [11] * 101
    best_mask = [0] * 101
    
    # Iterate through all non-empty subsets
    for mask in range(1, 1 << 10):
        # Calculate size and residue
        size = bin(mask).count('1')
        residue = 0
        # Unroll loop for speed? Or just iterate bits.
        # 10 bits is small.
        for i in range(10):
            if mask & (1 << i):
                residue = (residue + inv[i+1]) % 101
        
        # Update if this size is smaller for this residue
        if size < min_size[residue]:
            min_size[residue] = size
            best_mask[residue] = mask
            
    # The maximum row length is the max of min_size
    # max_k = max(min_size)
    
    # Construct the result
    result = []
    for r in range(101):
        mask = best_mask[r]
        # Decode mask to list of denominators
        terms = [i+1 for i in range(10) if mask & (1 << i)]
        result.append(terms)
        
    return result
