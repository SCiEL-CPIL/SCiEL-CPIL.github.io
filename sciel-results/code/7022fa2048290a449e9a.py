import time

def construct():
    """
    Find a, b such that a + b = 123457 and max(P(a), P(b)) is minimized.
    P(n) is the largest prime factor of n, with P(1)=1.
    Uses a sieve to precompute the largest prime factor for all numbers up to TARGET,
    then scans all pairs (a, TARGET-a) to find the pair minimizing max(P(a), P(b)).
    """
    TARGET = 123457
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin before 90s kill

    # Sieve for largest prime factor
    # lp[i] will store the largest prime factor of i
    lp = [0] * (TARGET + 1)
    lp[1] = 1  # P(1) = 1
    
    # Sieve of Eratosthenes variant: for each prime, mark its multiples
    # We iterate i from 2 to TARGET
    # If lp[i] == 0, i is prime, so we set lp for all multiples of i
    for i in range(2, TARGET + 1):
        if lp[i] == 0:
            # i is prime
            for j in range(i, TARGET + 1, i):
                lp[j] = i  # i is a prime factor of j, and since we go in increasing order, 
                           # the last prime that divides j will be the largest one
                           # Wait, this is wrong. If we iterate i from 2 upwards, 
                           # and for each prime i we update lp[j] = i, 
                           # then lp[j] will end up being the LARGEST prime factor of j.
                           # Because smaller primes are processed first, and larger primes overwrite them.
                           # Example: j=12. i=2: lp[12]=2. i=3: lp[12]=3. Correct, 3 is largest.
                           # Example: j=10. i=2: lp[10]=2. i=5: lp[10]=5. Correct.
    
    # Now lp[n] contains the largest prime factor for all n in [1, TARGET]
    
    # Scan for optimal pair
    best_val = float('inf')
    best_a = 1
    best_b = TARGET - 1
    
    for a in range(1, TARGET // 2 + 1):
        b = TARGET - a
        # We only need to check a <= b to avoid duplicates, but since we want min max,
        # checking all a from 1 to TARGET-1 is safe. However, due to symmetry max(P(a),P(b)) = max(P(b),P(a)),
        # and a+b is fixed, we can just scan a from 1 to TARGET//2.
        # Actually, just scan a from 1 to TARGET-1 to be safe and simple, or 1 to TARGET//2.
        # Let's scan 1 to TARGET//2.
        
        pa = lp[a]
        pb = lp[b]
        m = max(pa, pb)
        
        if m < best_val:
            best_val = m
            best_a = a
            best_b = b
            
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break
        
        # Early termination: if best_val is 1, it's optimal (only possible if both are 1, impossible for sum > 2)
        # If best_val is 2, it means both are powers of 2.
        # The theoretical minimum is small. We can't easily early terminate unless we hit a theoretical lower bound.
        # The smallest possible prime factor is 2. So min max is at least 2 (unless one is 1).
        # If a=1, b=123456. P(1)=1, P(123456)=? 123456 = 2^5 * 3 * 1029 = 2^5 * 3 * 3 * 343 = ... 
        # Anyway, we just scan. N is only 123457, so it's fast.

    return [best_a, best_b]
