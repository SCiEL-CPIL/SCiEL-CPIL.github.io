def construct():
    p = 1009
    # Precompute modular inverses for all elements 1..p-1
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    best_len = 0
    best_start = 1
    
    for s in range(1, p):
        # If the maximum possible length from s can't beat best_len, skip
        max_possible = p - s  # elements from s to p-1 = 1008
        if max_possible <= best_len:
            break
        
        for e in range(s, p):
            # Check diagonal: e^2 ≡ 1 mod p means e is 1 or p-1
            if e == 1 or e == p - 1:
                break
            
            # Check off-diagonal: is inv(e) already in [s, e-1]?
            if s <= inv[e] <= e - 1:
                break
            
            current_len = e - s + 1
            if current_len > best_len:
                best_len = current_len
                best_start = s
    
    return [best_start, best_len]
