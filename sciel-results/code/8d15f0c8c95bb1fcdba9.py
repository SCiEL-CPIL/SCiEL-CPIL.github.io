def construct():
    max_n = 1000000
    min_n = 10000
    
    # We need lpf for numbers up to max_n (since we check n+1 where n can be max_n-1, max value is max_n)
    limit = max_n
    lpf = [0] * (limit + 1)
    
    # Sieve to compute largest prime factor
    for i in range(2, limit + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, limit + 1, i):
                lpf[j] = i
                
    best_n = min_n
    best_val = float('inf')
    
    # Scan for the pair (n, n+1) minimizing max(lpf[n], lpf[n+1])
    for n in range(min_n, limit):
        lpf_n = lpf[n]
        lpf_n1 = lpf[n+1]
        current_max = lpf_n if lpf_n > lpf_n1 else lpf_n1
        
        if current_max < best_val:
            best_val = current_max
            best_n = n
            
    return best_n
