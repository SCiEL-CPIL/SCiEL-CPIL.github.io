import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
        
    # constraint[a] = b if b > a and a*b % MOD == 1, else INF
    # We want M[s] = min { b | a >= s, b > a, a*b % MOD == 1 }
    INF = N + 1
    constraint = [INF] * (N + 2)
    
    for a in range(1, N + 1):
        b = inv[a]
        if b > a:
            constraint[a] = b
            
    # Suffix minimums
    # M[s] = min(constraint[s], M[s+1])
    M = [INF] * (N + 2)
    M[N] = constraint[N]
    for s in range(N - 1, 0, -1):
        M[s] = min(constraint[s], M[s + 1])
        
    best_start = -1
    best_length = 0
    
    # s=1 is invalid because 1 is self-inverse
    # s > 1
    for s in range(2, N + 1):
        if time.time() > deadline:
            break
            
        # e must be < M[s] to avoid any pair (a, b) with a >= s, b <= e
        # e must be < 1008 to avoid 1008 being in the interval (since 1008 is self-inverse)
        e_limit = M[s] - 1
        e_cap = 1007
        e = min(e_limit, e_cap)
        
        if e >= s:
            length = e - s + 1
            if length > best_length:
                best_length = length
                best_start = s
                
    if best_length == 0:
        # Fallback, though unlikely
        return [2, 1]
        
    return [best_start, best_length]
