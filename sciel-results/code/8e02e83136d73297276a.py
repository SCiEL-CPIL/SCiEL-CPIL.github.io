"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Precompute modular inverses for all elements in [1, p-1]
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    best_len = 0
    best_start = 1
    
    # Self-inverse elements that can never be in a valid interval
    self_inverse = set([1, p - 1])
    
    for s in range(1, p):
        # Optimization: if max possible length from s is <= best_len, break
        if p - 1 - s + 1 <= best_len:
            break
        
        for e in range(s, p):
            # Check diagonal: e^2 == 1 mod p means e is self-inverse
            if e in self_inverse:
                break
            
            # Check off-diagonal: inv(e) must not be in [s, e-1]
            # (if inv(e) >= e, it's not in the interval yet since we're at e)
            if s <= inv[e] <= e - 1:
                break
            
            current_len = e - s + 1
            if current_len > best_len:
                best_len = current_len
                best_start = s
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
