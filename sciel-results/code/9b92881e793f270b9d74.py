"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    n = p - 1  # 1008
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin for 60s limit

    # Compute modular inverses
    # inv[x] = x^(-1) mod p
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        
    best_start = 2
    best_len = 0
    
    # Valid intervals must not contain 1 or 1008.
    # So a must be in [2, 1007] and b must be in [a, 1007].
    
    for a in range(2, 1008):
        if time.time() > deadline:
            break
            
        # Extend b as far as possible
        # We know [a, a] is valid if a is not self-inverse (which it isn't, as a in 2..1007).
        # Check if a is self-inverse: a*a % 1009 == 1?
        # Only 1 and 1008 are self-inverse. So [a, a] is always valid for a in 2..1007.
        
        current_len = 0
        for b in range(a, 1008):
            # Check if adding b to [a, b-1] creates a conflict.
            # Conflict exists if inv[b] is in [a, b].
            # Since inv[b] != b for b in 2..1007, this is equivalent to:
            # a <= inv[b] < b
            
            if inv[b] < a or inv[b] > b:
                # No conflict
                current_len = b - a + 1
            else:
                # Conflict found, stop extending
                break
                
        if current_len > best_len:
            best_len = current_len
            best_start = a
            
    return [best_start, best_len]
# EVOLVE-BLOCK-END
