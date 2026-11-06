def construct():
    MOD = 1009
    N = 1008
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
        
    best_start = 1
    best_len = 0
    s = 1
    
    for e in range(1, N + 1):
        ie = inv[e]
        if s <= ie <= e:
            s = ie + 1
        if s <= e:
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
                
    return [best_start, best_len]
