"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    MOD = 1009
    N = 1008
    
    # Precompute inverses
    inv = [0] * (MOD + 1)
    for i in range(1, MOD):
        # Fermat's little theorem: a^(p-2) % p
        # Or just loop, N is small
        # i * x = 1 mod 1009
        # Using pow is fast enough for 1008 items
        inv[i] = pow(i, MOD - 2, MOD)
        
    max_len = 0
    best_s = 1
    
    for s in range(1, MOD):
        if time.time() > deadline:
            break
            
        # Check if [s, s] is valid
        # Condition: inv[s] not in [s, s]
        if s <= inv[s] <= s:
            # inv[s] == s, so invalid
            continue
            
        current_e = s
        # Extend e
        for e in range(s + 1, MOD):
            if time.time() > deadline:
                break
                
            # Check if adding e makes interval [s, e] invalid
            # Invalid if inv[e] is in [s, e]
            if s <= inv[e] <= e:
                break
            current_e = e
            
        length = current_e - s + 1
        if length > max_len:
            max_len = length
            best_s = s
            
    return [best_s, max_len]
# EVOLVE-BLOCK-END
