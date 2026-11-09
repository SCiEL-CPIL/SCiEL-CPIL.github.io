def construct():
    LIMIT = 1009
    MAX_VAL = 1008
    MIN_VAL = 2
    MAX_VALID = 1007

    # Precompute modular inverses
    inv = [0] * (LIMIT + 1)
    for i in range(1, LIMIT):
        inv[i] = pow(i, -1, LIMIT)

    # For each x where inv[x] > x, the pair (x, inv[x]) conflicts.
    # For a fixed L, R_max(L) = min(MAX_VALID, min{inv[x]-1 : x >= L, inv[x] > x})
    # This is a suffix minimum problem.
    
    INF = float('inf')
    # arr[x] = inv[x] - 1 if inv[x] > x, else INF
    arr = [INF] * (MAX_VALID + 2)
    for x in range(MIN_VAL, MAX_VALID + 1):
        if inv[x] > x:
            arr[x] = inv[x] - 1
    
    # suffix_min[x] = min(arr[x], arr[x+1], ..., arr[MAX_VALID])
    suffix_min = [INF] * (MAX_VALID + 2)
    for x in range(MAX_VALID, MIN_VAL - 1, -1):
        suffix_min[x] = min(arr[x], suffix_min[x + 1])
    
    # For each L, compute max valid R and track best length
    best_len = 0
    best_start = MIN_VAL
    
    for L in range(MIN_VAL, MAX_VALID + 1):
        r_max = min(MAX_VALID, suffix_min[L])
        if r_max >= L:
            length = r_max - L + 1
            if length > best_len:
                best_len = length
                best_start = L
    
    if best_len == 0:
        best_start = MIN_VAL
        best_len = 1
    
    return [best_start, best_len]
