import time

def construct():
    MOD = 1009
    N = 1008
    start_time = time.time()
    deadline = 80  # safety margin

    # Precompute inverses
    inv = [0] * (MOD)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)

    best_start = 2
    best_len = 0

    # Iterate over possible starting positions L
    # L cannot be 1 or 1008 because they are self-inverse
    for L in range(2, 1008):
        if time.time() - start_time > deadline:
            break
        
        # Find M = min { inv[x] | x in [L, 1008] and inv[x] >= L }
        # If no such x exists, M is effectively infinity
        M = 1009 
        
        # We only need to check x in [L, 1008]
        # Note: if inv[x] < L, it doesn't constrain R from above (it's already < L)
        for x in range(L, 1009):
            ix = inv[x]
            if ix >= L:
                if ix < M:
                    M = ix
        
        if M == 1009:
            R = 1008
        else:
            R = M - 1
        
        if R >= L:
            length = R - L + 1
            if length > best_len:
                best_len = length
                best_start = L

    return [best_start, best_len]
