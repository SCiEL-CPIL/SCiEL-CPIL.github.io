"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0
    p = 1009
    
    # Precompute modular inverses
    inv = [0] * (p + 1)
    self_inv = [False] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        if x * x % p == 1:
            self_inv[x] = True
    
    best_start = 2
    best_len = 1
    
    for s in range(1, 1009):
        if time.time() > deadline:
            break
        
        if 1008 - s + 1 <= best_len:
            continue
            
        if self_inv[s]:
            continue
        
        e = s
        while e < 1008:
            new_e = e + 1
            if self_inv[new_e]:
                break
            if s <= inv[new_e] <= new_e:
                break
            e = new_e
        
        length = e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
