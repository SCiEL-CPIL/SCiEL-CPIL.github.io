import time

def construct():
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    best_start = 1
    best_len = 0
    
    # Optional: Deadline guard, though O(N^2) is very fast
    start_time = time.time()
    deadline = start_time + 50.0  # 50s margin
    
    for L in range(1, N + 1):
        if time.time() > deadline:
            break
            
        # If L is self-inverse, it cannot be in any valid interval
        if inv[L] == L:
            continue
            
        R = L
        # We extend R as far as possible
        # [L, R] is valid if for all x in [L, R], inv[x] is not in [L, R]
        # If [L, R-1] is valid, then [L, R] is valid iff inv[R] is not in [L, R]
        # because any conflict involving R must be with R itself or with some x in [L, R-1].
        # If conflict with R itself: inv[R] == R.
        # If conflict with x: inv[R] == x. Since x in [L, R-1], this means L <= inv[R] <= R-1.
        # So combined: conflict iff L <= inv[R] <= R.
        
        valid = True
        while R <= N and valid:
            if L <= inv[R] <= R:
                valid = False
            else:
                current_len = R - L + 1
                if current_len > best_len:
                    best_len = current_len
                    best_start = L
                R += 1
                # Check time in inner loop if necessary, but N is small
                if time.time() > deadline:
                    break
                    
    return [best_start, best_len]
