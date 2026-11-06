"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    n = p - 1  # 1008
    
    # Compute modular inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # Precompute self-inverse flags
    is_self_inv = [False] * (p + 1)
    for x in range(1, p):
        if (x * x) % p == 1:
            is_self_inv[x] = True
            
    best_start = 2
    best_len = 0
    
    for s in range(1, n + 1):
        if is_self_inv[s]:
            continue
            
        e = s
        for j in range(s + 1, n + 1):
            if is_self_inv[j]:
                break
            if s <= inv[j] < j:
                break
            e = j
            
        length = e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
            
    return [best_start, best_len]
# EVOLVE-BLOCK-END
