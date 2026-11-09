def construct():
    import time
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin within 90s limit
    
    MOD = 1009
    N = 1008
    
    # Compute modular inverses for 1..1008
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Valid interval must be within [2, 1007] (exclude self-inverse 1 and 1008)
    lo = 2
    hi = 1007
    
    best_start = lo
    best_len = 0
    
    left = lo
    for right in range(lo, hi + 1):
        inv_r = inv[right]
        # inv_r is in [1, 1008]. For x in [2,1007], inv[x] != x.
        # Check if inv_r is in the current window [left, right]
        if lo <= inv_r <= hi and inv_r >= left and inv_r < right:
            # Conflict detected: both right and inv_r would be in the window
            left = inv_r + 1
        
        if left < lo:
            left = lo
        
        current_len = right - left + 1
        if current_len > best_len:
            best_len = current_len
            best_start = left
    
    # Verify the result is valid
    for x in range(best_start, best_start + best_len):
        for y in range(best_start, best_start + best_len):
            if (x * y) % MOD == 1:
                # Fallback: return a known valid small interval
                return [2, 1]
    
    return [best_start, best_len]
