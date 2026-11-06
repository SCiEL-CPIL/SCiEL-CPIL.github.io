def construct():
    mod = 1009
    # Compute modular inverses in O(N)
    # inv[i] = (mod - mod//i) * inv[mod%i] % mod
    inv = [0] * (mod + 1)
    inv[1] = 1
    for i in range(2, mod + 1):
        inv[i] = (mod - mod // i) * inv[mod % i] % mod
    
    best_start = 1
    best_len = 1
    
    for s in range(1, mod):
        # Find the minimum max(x, inv[x]) for all x >= s such that inv[x] >= s
        # This is the first point where the interval [s, e] becomes invalid
        min_block = mod + 1 # Infinity
        
        for x in range(s, mod):
            if inv[x] >= s:
                block = max(x, inv[x])
                if block < min_block:
                    min_block = block
        
        # The interval [s, e] is valid for e < min_block
        # So max valid e is min_block - 1
        # But e cannot exceed mod - 1 (which is 1008)
        # If no block found, min_block is mod + 1, so e = mod = 1009? No, max index is 1008.
        # So e_valid = min(min_block - 1, mod - 1)
        
        e_valid = min(min_block - 1, mod - 1)
        
        # Ensure e_valid >= s (interval is non-empty)
        if e_valid < s:
            continue
            
        length = e_valid - s + 1
        if length > best_len:
            best_len = length
            best_start = s
    
    return [best_start, best_len]
