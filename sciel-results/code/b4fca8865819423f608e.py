import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit
    
    MOD = 1009
    N = 1008  # max value in [1, 1008]
    
    # Precompute modular inverses for all x in [1, N]
    # inv[x] is the unique y in [1, N] such that x*y % MOD == 1
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    # Identify self-inverse elements: x where x*x % MOD == 1
    # These are x=1 and x=MOD-1=1008
    # They cannot be in any valid interval (diagonal check fails)
    
    best_start = 1
    best_length = 0
    
    # For each position i, precompute the nearest conflicting partner to the left
    # conflict_left[i] = max { j < i : inv[j] == i or inv[i] == j } if exists, else 0
    # Actually, we need: for a given start L, what is the earliest R where adding R creates a conflict?
    # A conflict at R means inv[R] is in [L, R].
    
    # Let's precompute for each value v, its inverse inv[v].
    # For a fixed start L, we can extend R as long as inv[R] < L or inv[R] > R.
    # Wait, inv[R] > R is fine only if inv[R] is not in [L,R]. Since inv[R] > R, it's not in [L,R].
    # So the condition for R to be valid given start L is: inv[R] < L.
    # (Because inv[R] == R is impossible for R in [2, 1007] since only 1 and 1008 are self-inverse)
    
    # So for a fixed L, the maximum R is the largest integer such that for all k in [L, R], inv[k] < L.
    # This is equivalent to: R is the largest integer < first_bad where first_bad is the smallest k >= L with inv[k] >= L.
    # Note: inv[k] >= L includes inv[k] in [L, k] (conflict) or inv[k] > k (which is > k >= L, so inv[k] >= L is true but not a conflict).
    # Wait, my previous logic: conflict if L <= inv[R] <= R.
    # If inv[R] > R, then inv[R] is NOT in [L, R]. So no conflict.
    # If inv[R] < L, then inv[R] is NOT in [L, R]. So no conflict.
    # If L <= inv[R] <= R, conflict.
    
    # So the condition to STOP at R is: L <= inv[R] <= R.
    # This is equivalent to: inv[R] >= L AND inv[R] <= R.
    
    # Let's precompute an array 'min_inv_from_right' or similar?
    # Actually, for each L, we just need to find the first R >= L such that inv[R] >= L and inv[R] <= R.
    # Since R increases, inv[R] <= R is a condition on R.
    
    # Let's just iterate. The total work is sum over L of (R_max - L + 1).
    # In the worst case, if no conflicts, R_max = N for all L, total work ~ N^2/2 ~ 500,000.
    # This is very fast.
    
    for L in range(1, N + 1):
        if time.time() > deadline:
            break
        
        # If L is self-inverse, length is 0 (or we skip, but length 0 is not nonempty)
        # The problem says "nonempty interval", so length >= 1.
        # If L is 1 or 1008, we cannot even start with length 1.
        # But the loop goes through them, R will be L-1, length 0.
        
        R = L - 1
        while R < N:
            if time.time() > deadline:
                break
            
            next_R = R + 1
            # Check if next_R can be added
            # Conflict if L <= inv[next_R] <= next_R
            if L <= inv[next_R] <= next_R:
                break
            R = next_R
        
        current_length = R - L + 1
        if current_length > best_length:
            best_length = current_length
            best_start = L
    
    # If no valid interval found (should not happen), return a default
    if best_length == 0:
        return [1, 1]  # Fallback, though 1 is self-inverse so this would be invalid.
                       # But 1*1 % 1009 = 1, so [1,1] is invalid.
                       # However, there must be valid intervals. e.g., [2, 2] is valid since 4 != 1.
    
    return [best_start, best_length]
