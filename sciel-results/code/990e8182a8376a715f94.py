import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    p = 1009
    n = 1008
    
    # Precompute modular inverses mod p for x in [1, p-1]
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)
        
    # Identify self-inverse elements: x^2 = 1 mod p
    # For prime p, x = 1 or x = p-1
    self_inv = {1, p - 1}
    
    # Two-pointer approach to find the longest valid interval
    # Window is [L, R]
    L = 1
    present = [0] * (n + 1)  # present[x] is 1 if x is in [L, R]
    conflict_count = 0
    
    best_start = 2
    best_length = 0
    
    for R in range(1, n + 1):
        if time.time() > deadline:
            break
            
        # Add R to the window
        present[R] = 1
        
        if R in self_inv:
            conflict_count += 1
        else:
            inv_R = inv[R]
            # Check if the inverse of R is already in the window
            # Since R is being added, inv_R must be < R or > R.
            # If inv_R is in [L, R-1], it's a conflict.
            if present[inv_R]:
                conflict_count += 1
        
        # Shrink window from the left until it is valid
        while conflict_count > 0 and L <= R:
            # Remove L from the window
            if L in self_inv:
                conflict_count -= 1
            else:
                inv_L = inv[L]
                # If the inverse of L is still in the window (i.e., > L), 
                # then removing L resolves the conflict created by that pair.
                # Note: if inv_L < L, the conflict was already resolved when 
                # inv_L was removed, or it wasn't a conflict with L as the 
                # "new" element in the first place? 
                # Actually, the conflict is counted when the SECOND element of 
                # the pair is added. 
                # If we added R and R's inverse was in [L, R-1], conflict++.
                # If we remove L, and L was the inverse of some element Y in (L, R],
                # then the pair {L, Y} is broken.
                # So, if present[inv_L] is 1, it means inv_L is in (L, R], 
                # so the pair {L, inv_L} existed and contributed to conflict_count.
                if present[inv_L]:
                    conflict_count -= 1
            
            present[L] = 0
            L += 1
        
        # Now [L, R] is a valid interval
        current_length = R - L + 1
        if current_length > best_length:
            best_length = current_length
            best_start = L

    # If no valid interval found (should not happen, single elements like 2 are valid), 
    # fallback to a known valid single element
    if best_length == 0:
        best_start = 2
        best_length = 1

    # Verification step to ensure correctness
    # Check [best_start, best_start + best_length - 1]
    end = best_start + best_length - 1
    valid = True
    for x in range(best_start, end + 1):
        if x in self_inv:
            valid = False
            break
        ix = inv[x]
        if best_start <= ix <= end:
            valid = False
            break
            
    if not valid:
        # Fallback: brute force search for any valid interval (should rarely be needed)
        # Try to find a valid interval of length 1, then expand
        for s in range(1, n + 1):
            if s in self_inv:
                continue
            # Check if [s, s] is valid
            # It is, unless s is self-inverse (already checked)
            # Try to extend
            e = s
            while e < n:
                next_e = e + 1
                if next_e in self_inv:
                    break
                if best_start <= inv[next_e] <= next_e: # Wait, this check is local
                    # Need to check against whole [s, next_e]
                    # For simplicity in fallback, just check if inv[next_e] is in [s, next_e]
                    # And if inv[s..e] contains next_e
                    # This fallback is just a safety net.
                    pass
                # Simplified fallback: just return a single element
                break
            best_start = s
            best_length = 1
            break

    return [best_start, best_length]
