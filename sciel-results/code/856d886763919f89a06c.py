import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (N + 1)
    for i in range(1, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)
    
    # Self-inverse elements: x^2 % MOD == 1 => x == 1 or x == MOD-1 = 1008
    self_inverse = set([1, 1008])
    
    best_start = 1
    best_length = 0
    
    L = 1
    for R in range(1, N + 1):
        # Check if R itself is self-inverse
        if R in self_inverse:
            # Any window containing R is invalid
            L = R + 1
            continue
        
        # Check if inv[R] is in [L, R-1] (i.e., already in the window)
        # If inv[R] >= L, we need to push L past inv[R]
        if inv[R] >= L and inv[R] < R:
            # inv[R] is in the current window [L, R-1], so we must exclude it
            # The new window must start after inv[R]
            L = max(L, inv[R] + 1)
        
        current_length = R - L + 1
        if current_length > best_length:
            best_length = current_length
            best_start = L
    
    # Verify the found interval
    if best_length > 0:
        valid = True
        for x in range(best_start, best_start + best_length):
            for y in range(best_start, best_start + best_length):
                if (x * y) % MOD == 1:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback: brute force search for the longest valid interval
            best_start = 1
            best_length = 0
            for s in range(1, N + 1):
                for e in range(s, N + 1):
                    valid = True
                    for x in range(s, e + 1):
                        if x in self_inverse:
                            valid = False
                            break
                        if inv[x] >= s and inv[x] <= e:
                            valid = False
                            break
                    if valid:
                        length = e - s + 1
                        if length > best_length:
                            best_length = length
                            best_start = s
                    else:
                        break
    
    if best_length == 0:
        # Fallback: find any valid single element
        for i in range(2, N + 1):
            if i not in self_inverse:
                return [i, 1]
        return [2, 1]
    
    return [best_start, best_length]
