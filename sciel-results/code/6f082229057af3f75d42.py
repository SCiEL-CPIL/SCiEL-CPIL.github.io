import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = 90.0
    
    # 1. Compute Largest Prime Factor for all numbers up to N
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for i in range(2, N + 1):
        if lpf[i] == 0: # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    # 2. Find the split a+b=N minimizing max(P(a), P(b))
    best_a = 1
    best_b = N - 1
    best_max_p = lpf[best_a] if best_a > 1 else 1
    best_max_p = max(best_max_p, lpf[best_b] if best_b > 1 else 1)
    
    # We only need to check up to N//2 because a,b are symmetric
    for a in range(2, N // 2 + 1):
        b = N - a
        p_a = lpf[a]
        p_b = lpf[b]
        current_max = max(p_a, p_b)
        
        if current_max < best_max_p:
            best_max_p = current_max
            best_a = a
            best_b = b
            
        # Early termination check? 
        # The theoretical minimum for max(P(a), P(b)) is at least 2 (if both are powers of 2, but sum is odd, so one must be odd).
        # Actually, if N is odd, one is even, one is odd.
        # If we find a split where max(P(a), P(b)) is small, can we stop?
        # Not easily without knowing the global lower bound.
        # But the loop is O(N) ~ 60k iterations, which is instant.
        
        if time.time() > deadline - 5.0:
            break
            
    return [best_a, best_b]
