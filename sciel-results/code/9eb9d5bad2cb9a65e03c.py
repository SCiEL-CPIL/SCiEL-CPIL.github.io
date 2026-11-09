import time

def construct():
    p = 1009
    start_time = time.time()
    
    # Precompute modular inverses
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
        
    # A[a] = min b such that min(x, inv[x]) == a
    # We want M(L) = min { b_x | min(x, inv[x]) >= L }
    # Let A[a] be the min b for pairs starting at a.
    # Then M(L) = min( A[L], A[L+1], ..., A[p-1] )
    
    INF = float('inf')
    A = [INF] * p
    
    for x in range(1, p):
        u = x
        v = inv[x]
        if u > v:
            u, v = v, u
        # u is the smaller element, v is the larger
        # If u >= L, then v is a constraint on R.
        # We store min v for each u.
        if v < A[u]:
            A[u] = v
            
    # Compute suffix minimums
    # suf_min[i] = min(A[i], A[i+1], ..., A[p-1])
    suf_min = [INF] * p
    current_min = INF
    for i in range(p - 1, -1, -1):
        if A[i] < current_min:
            current_min = A[i]
        suf_min[i] = current_min
        
    best_len = 0
    best_start = 1
    
    for L in range(1, p):
        # Max valid R is suf_min[L] - 1
        R = suf_min[L] - 1
        if R < L:
            continue
        # Check if interval [L, R] is within [1, p-1]
        # L >= 1 is guaranteed by loop.
        # R <= p-1 is guaranteed because suf_min[L] <= p-1 (since max v is p-1)
        # Wait, max v is p-1=1008. So R <= 1007.
        # So [L, R] is always within [1, 1007].
        # Self-inverses 1 and 1008 are handled:
        # If L=1, A[1] has pair (1,1). suf_min[1] <= 1. R <= 0. R < L. Skipped.
        # If L<=1008, pair (1008,1008) has u=1008.
        # If L <= 1008, A[1008] <= 1008.
        # suf_min[L] <= A[1008] <= 1008.
        # R <= 1007. So 1008 is not in [L, R].
        
        length = R - L + 1
        if length > best_len:
            best_len = length
            best_start = L
            
    return [best_start, best_len]
