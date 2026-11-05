import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Primes p ≡ 3 mod 4 in range
    # 31, 43, 47, 59, 67, 71, 79, 83
    candidates = [31, 43, 47, 59, 67, 71, 79, 83]
    
    best_n = 83 # Fallback to seed
    best_orientation = None
    
    # Generate orientation for seed 83 just in case we need to return it
    # Actually, we can just construct it on the fly if we fail all others
    
    for p in candidates:
        if time.time() > deadline:
            break
            
        # Check if p is prime (hardcoded list is fine for this small range)
        # Construct Paley
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
        
        # Build out_masks
        # out_masks[i] is an int with bit j set if i -> j
        out_masks = [0] * p
        for i in range(p):
            mask = 0
            for j in range(p):
                if i == j: continue
                if (j - i) % p in squares:
                    mask |= (1 << j)
            out_masks[i] = mask
            
        # Check validity
        valid = True
        # Iterate over all 4-combinations
        # Optimization: If n is large, this is the bottleneck.
        # For p=31, ~30k combos.
        # For p=67, ~700k combos.
        
        # We can break early if invalid
        for combo in itertools.combinations(range(p), 4):
            m = out_masks[combo[0]] & out_masks[combo[1]] & out_masks[combo[2]] & out_masks[combo[3]]
            if m == 0:
                valid = False
                break
                
        if valid:
            best_n = p
            # Construct orientation list for output
            # Format: one bit per pair (i,j), i<j, lexicographic.
            # 0 means i -> j, 1 means j -> i.
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    # i -> j iff (j-i)%p in squares
                    if (j - i) % p in squares:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            
            # Return immediately as we want minimum n
            return {'n': p, 'orientation': orientation}
            
    # If no valid found in loop (should not happen for 83), return seed
    # Re-construct seed 83
    p = 83
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
                
    return {'n': p, 'orientation': orientation}
