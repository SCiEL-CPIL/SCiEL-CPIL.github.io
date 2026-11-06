"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    p = 1009
    n = p - 1  # 1008
    
    # Compute modular inverses
    # inv[x] = x^(p-2) mod p
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        
    best_start = 2
    best_len = 0
    
    # Iterate over all possible starts
    for a in range(1, n + 1):
        # Check time
        if time.time() > deadline:
            break
            
        # Skip self-inverse residues
        if a == 1 or a == 1008:
            continue
            
        # Interval [a, a] is valid (since a is not 1 or 1008)
        current_max_end = a
        current_len = 1
        
        # Try to extend the interval
        for b in range(a + 1, n + 1):
            # Check time
            if time.time() > deadline:
                break
                
            # If b is self-inverse, it cannot be in the interval
            if b == 1 or b == 1008:
                break
                
            # Check if inv[b] is in [a, b-1]
            # If so, there exists x in [a, b-1] such that x * b == 1
            if a <= inv[b] <= b - 1:
                break
            
            # If we reach here, [a, b] is valid
            current_max_end = b
            current_len = b - a + 1
            
        if current_len > best_len:
            best_len = current_len
            best_start = a
            
    return [best_start, best_len]
# EVOLVE-BLOCK-END
