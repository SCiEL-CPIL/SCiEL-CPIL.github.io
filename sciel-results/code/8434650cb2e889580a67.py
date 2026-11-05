import math

def construct():
    MAX_N = 1000000
    LIMIT = MAX_N + 1
    
    # Sieve for largest prime factor
    lpf = [0] * (LIMIT + 1)
    for i in range(2, LIMIT + 1):
        if lpf[i] == 0: # i is prime
            for j in range(i, LIMIT + 1, i):
                lpf[j] = i
                
    best_n = 10000
    min_lpf_val = float('inf')
    
    for n in range(10000, MAX_N + 1):
        # n and n+1 are both >= 2, so lpf is defined
        l1 = lpf[n]
        l2 = lpf[n+1]
        current_max = l1 if l1 > l2 else l2
        
        if current_max < min_lpf_val:
            min_lpf_val = current_max
            best_n = n
            
    return best_n
