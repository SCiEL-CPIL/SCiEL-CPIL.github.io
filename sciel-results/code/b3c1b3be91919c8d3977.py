import time

def construct():
    deadline = time.time() + 85.0
    
    p = 101
    # Inverses of 1..10 mod 101
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, p)
        
    # min_size[r] will store the minimum subset size achieving residue r
    # Initialize with a large number (11)
    min_size = [11] * p
    
    # Iterate over all non-empty subsets of {1..10}
    # There are 2^10 - 1 = 1023 subsets
    # We can iterate using bitmask or itertools
    # Bitmask is fast
    for mask in range(1, 1 << 10):
        # Calculate sum of inverses and size
        s = 0
        size = 0
        # Unroll loop for speed? Or just simple loop
        for i in range(10):
            if mask & (1 << i):
                s += inv[i + 1]
                size += 1
        s %= p
        if size < min_size[s]:
            min_size[s] = size
            
    # Check if all residues are covered
    # If any min_size is 11, it means not covered (should not happen for size <= 10)
    max_len = max(min_size)
    
    # Now construct the rows
    # We need to pick a specific subset for each residue
    # We can just pick any subset of size min_size[r]
    # To do this, we need to store one example subset for each (residue, size) or just residue
    # Since we want to minimize max length, and we already found the min length,
    # we just need to find ONE subset of that length for each residue.
    
    # Re-scan to find the actual subsets
    best_subset = [None] * p
    
    for mask in range(1, 1 << 10):
        s = 0
        size = 0
        for i in range(10):
            if mask & (1 << i):
                s += inv[i + 1]
                size += 1
        s %= p
        
        # If this subset size matches the minimum required for this residue,
        # and we haven't picked one yet, pick it.
        if size == min_size[s] and best_subset[s] is None:
            # Extract denominators
            subset = []
            for i in range(10):
                if mask & (1 << i):
                    subset.append(i + 1)
            best_subset[s] = subset
            
    # Verify all are found
    for r in range(p):
        if best_subset[r] is None:
            # Fallback: find any subset
            # This should not happen
            for mask in range(1, 1 << 10):
                s = 0
                size = 0
                for i in range(10):
                    if mask & (1 << i):
                        s += inv[i + 1]
                        size += 1
                s %= p
                if s == r:
                    subset = []
                    for i in range(10):
                        if mask & (1 << i):
                            subset.append(i + 1)
                    best_subset[r] = subset
                    break
                    
    return best_subset
