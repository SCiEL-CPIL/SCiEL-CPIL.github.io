import time

def construct():
    # The problem is small enough that we can afford O(N^2)
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses for 1 to 1008
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
        
    max_len = 0
    best_start = 1
    
    # Try every possible start index
    for s in range(1, N + 1):
        # Find the maximum valid end index for this start
        # An interval [s, e] is valid if for all x in [s, e], inv[x] is not in [s, e]
        # We can extend e incrementally.
        # Adding e to [s, e-1] is valid iff inv[e] is not in [s, e]
        # which is equivalent to NOT (s <= inv[e] <= e)
        
        valid_end = s - 1
        for e in range(s, N + 1):
            if s <= inv[e] <= e:
                break
            valid_end = e
            
        current_len = valid_end - s + 1
        if current_len > max_len:
            max_len = current_len
            best_start = s
            
    return [best_start, max_len]
