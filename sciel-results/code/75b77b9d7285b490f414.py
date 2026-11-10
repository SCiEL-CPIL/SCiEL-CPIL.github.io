import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for evaluator overhead
    
    # Upper bound for values we need LPF for is 1000000 + 1 = 1000001
    max_val = 1000001
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (max_val + 1)
    
    # Sieve to find largest prime factors
    # For each prime p, it is the largest prime factor for all multiples of p
    # that are not divisible by any prime > p.
    # Since we iterate p in increasing order, if we simply assign lpf[j] = p 
    # for all multiples j, the last assignment (largest p) will remain.
    for p in range(2, max_val + 1):
        if lpf[p] == 0:
            # p is prime
            for j in range(p, max_val + 1, p):
                lpf[j] = p
        
        # Time check every 50000 iterations
        if p % 50000 == 0 and time.time() > deadline:
            return 10000
    
    # Time check
    if time.time() > deadline:
        return 10000

    best_n = 10000
    min_score = float('inf')
    
    # Scan for the n that minimizes max(lpf[n], lpf[n+1])
    # Use local variable for lpf for faster access
    lpf_local = lpf
    for n in range(10000, 1000001):
        # The score is the largest prime factor of the product n(n+1)
        # which is max(lpf[n], lpf[n+1])
        a = lpf_local[n]
        b = lpf_local[n + 1]
        score = a if a >= b else b
        
        if score < min_score:
            min_score = score
            best_n = n
            
            # Early termination: if we find a very low score
            if min_score <= 5:
                break
    
    return best_n
