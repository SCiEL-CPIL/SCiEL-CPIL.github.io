import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Primes p = 3 mod 4
    primes = [3, 7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79, 83]
    
    # Filter for n >= 5
    candidates = [p for p in primes if p >= 5]
    
    # We want the smallest n, so we iterate in increasing order of p
    # But we know 83 is valid (from seed). If a smaller one is valid, we return it.
    
    for p in candidates:
        if time.time() > deadline:
            break
            
        # Construct Paley tournament
        # i -> j iff (j - i) is a quadratic residue mod p
        # Note: 0 is not a residue.
        squares = {x * x % p for x in range(1, p)}
        
        # Precompute in-masks
        # in_masks[v] has bit u set if u -> v
        in_masks = [0] * p
        # For each pair i, j, determine direction
        # If i -> j, then j is in out-neighbors of i, and i is in in-neighbors of j.
        # So bit i is set in in_masks[j].
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in squares:
                    # i -> j
                    in_masks[j] |= (1 << i)
                else:
                    # j -> i
                    in_masks[i] |= (1 << j)
                    
        # Check validity
        # Every 4-subset must have a common dominator outside the subset
        valid = True
        all_ones = (1 << p) - 1
        
        # Optimization: if p is large, this loop is heavy.
        # For p=83, ~2.1M combos.
        # For p=67, ~7.3M combos.
        # Python loop might be slow for 7.3M.
        # Let's estimate: 10M ops/sec is optimistic for complex Python loops.
        # 7.3M iterations * ~50ns/iter? No, Python is ~100ns-1us per simple op.
        # Bitwise ops are C-speed, but loop overhead is Python.
        # 7.3M * 1us = 7.3 seconds.
        # Acceptable.
        
        for quad in itertools.combinations(range(p), 4):
            v1, v2, v3, v4 = quad
            common = in_masks[v1] & in_masks[v2] & in_masks[v3] & in_masks[v4]
            mask_s = (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
            # Check if there is a dominator outside the set
            if not (common & (~mask_s & all_ones)):
                valid = False
                break
                
        if valid:
            # Found a valid tournament with n=p
            # Generate orientation list
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    if (j - i) % p in squares:
                        orientation.append(0) # i -> j
                    else:
                        orientation.append(1) # j -> i
            return {'n': p, 'orientation': orientation}
            
    # Fallback to 83 if loop breaks due to time or no smaller found
    p = 83
    squares = {x * x % p for x in range(1, p)}
    orientation = [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]
    return {'n': p, 'orientation': orientation}
