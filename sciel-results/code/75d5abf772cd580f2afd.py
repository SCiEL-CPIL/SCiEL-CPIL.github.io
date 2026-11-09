import time

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    N_MAX = 1000000
    MAX_VAL = N_MAX + 24
    
    # Precompute the small part for all numbers up to MAX_VAL
    # small_part[x] will be the factor of x composed of primes in 'primes'
    small_part = [1] * (MAX_VAL + 1)
    
    # Sieve-like update: for each prime p, multiply p into small_part[m]
    # for each power p^k dividing m.
    for p in primes:
        power = p
        while power <= MAX_VAL:
            for multiple in range(power, MAX_VAL + 1, power):
                small_part[multiple] *= p
            power *= p
            
    best_n = 1000
    min_distinct = 100  # Start high
    
    start_time = time.time()
    deadline = 85  # Leave 5s margin for safety
    
    sp = small_part
    
    # Search for n that minimizes distinct small factors in n+1..n+24
    for n in range(1000, N_MAX + 1):
        # Check time every 1024 iterations to reduce overhead
        if (n & 1023) == 0 and time.time() - start_time > deadline:
            break
            
        # Extract the window and count distinct values
        # Slicing and set conversion is implemented in C and is fast
        factors = set(sp[n+1 : n+25])
        distinct_count = len(factors)
        
        if distinct_count < min_distinct:
            min_distinct = distinct_count
            best_n = n
            
            if min_distinct == 1:
                break

    return best_n

construct()
