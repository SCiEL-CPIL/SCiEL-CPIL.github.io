def construct():
    N = 123457
    
    # Sieve for largest prime factor
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (N + 1)
    lpf[1] = 1
    
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    best_val = float('inf')
    best_pair = [1, N - 1]
    
    # Only need to check up to N//2
    for a in range(1, N // 2 + 1):
        b = N - a
        val = lpf[a] if lpf[a] > lpf[b] else lpf[b]
        if val < best_val:
            best_val = val
            best_pair = [a, b]
            # Optional: early termination if best_val is 2 (theoretical min for sum > 2)
            # But 2 is impossible as shown. 3 is impossible. 
            # So we just run through.
            
    return best_pair
