def construct():
    max_val = 1000000
    min_n = 10000
    
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (max_val + 1)
    
    # Sieve to find largest prime factors
    # We iterate i from 2 to max_val.
    # If lpf[i] is 0, i is prime.
    # Then for j in range(i, max_val+1, i):
    #     lpf[j] = i
    # Since i increases, the last assignment is the largest prime factor.
    
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, max_val + 1, i):
                lpf[j] = i
                
    # Now find n in [min_n, max_val - 1] that minimizes max(lpf[n], lpf[n+1])
    best_n = min_n
    best_score = float('inf')
    
    for n in range(min_n, max_val):
        score = lpf[n]
        if lpf[n+1] > score:
            score = lpf[n+1]
            
        if score < best_score:
            best_score = score
            best_n = n
            
    return best_n
