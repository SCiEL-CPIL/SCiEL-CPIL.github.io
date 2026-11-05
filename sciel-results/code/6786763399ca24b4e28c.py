import time

def construct():
    """
    Find a, b such that a + b = 123457 and max(P(a), P(b)) is minimized.
    P(n) is the largest prime factor of n, with P(1)=1.
    
    Strategy:
    1. Precompute the largest prime factor (LPF) for all numbers up to TARGET using a sieve.
    2. Iterate through possible values of a from 1 to TARGET//2.
    3. For each a, compute b = TARGET - a.
    4. Track the pair that minimizes max(LPF[a], LPF[b]).
    5. Since N=123457 is small, this O(N) scan after O(N log log N) sieve is very fast (< 1 ms).
    """
    TARGET = 123457
    start_time = time.time()
    DEADLINE = 85  # seconds

    # 1. Sieve for Largest Prime Factor (LPF)
    # lpf[i] will store the largest prime factor of i.
    # For primes, lpf[i] = i.
    # For composites, it will be updated to the largest prime divisor.
    
    # Initialize lpf array
    # We only need to go up to TARGET.
    lpf = [0] * (TARGET + 1)
    
    # lpf[1] = 1 by definition (P(1)=1)
    lpf[1] = 1
    
    # Standard sieve variation to find largest prime factors
    # For each prime p, mark multiples of p.
    # To get the LARGEST prime factor, we should iterate primes in increasing order
    # and update lpf[multiple] = p. Since we go from small to large primes,
    # the last prime that divides a number will be the largest.
    # Wait, that's not quite right.
    # If we iterate p from 2 to sqrt(TARGET) (or all primes), and for each multiple m of p,
    # we set lpf[m] = p, then for a number like 12 = 2*2*3:
    # p=2: lpf[4]=2, lpf[6]=2, lpf[8]=2, lpf[10]=2, lpf[12]=2...
    # p=3: lpf[6]=3, lpf[9]=3, lpf[12]=3...
    # So lpf[12] becomes 3. Correct.
    # What about a prime p > sqrt(TARGET)?
    # If n is prime, it won't be touched by any smaller prime.
    # We need to ensure primes are identified.
    # A better way:
    # 1. Initialize lpf[i] = i for all i? No, that's O(N).
    # 2. Standard approach:
    #    lpf = [0]*(N+1)
    #    for i in range(2, N+1):
    #        if lpf[i] == 0: # i is prime
    #            lpf[i] = i
    #            for j in range(i, N+1, i):
    #                if lpf[j] == 0:
    #                    lpf[j] = i # This gives SMALLEST prime factor
    #
    #    To get LARGEST prime factor, we can do:
    #    lpf = [0]*(N+1)
    #    for i in range(2, N+1):
    #        if lpf[i] == 0: # i is prime
    #            for j in range(i, N+1, i):
    #                lpf[j] = i # Overwrite with larger prime as we go up
    #    After this loop, lpf[x] will be the largest prime factor of x.
    #    Why? Because we iterate i from 2 to N. If i is prime, we update all its multiples.
    #    Since i increases, the last update to lpf[j] comes from the largest prime divisor of j.
    #    Note: This is O(N log log N) roughly, sum of N/p for primes p.
    
    # Let's implement this efficient sieve.
    for i in range(2, TARGET + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, TARGET + 1, i):
                lpf[j] = i
    
    # lpf[1] is already 1.
    # Now lpf[n] holds the largest prime factor of n for all n in 1..TARGET.
    
    # 2. Scan for optimal pair
    best_max_p = TARGET  # Initial upper bound (worst case is a prime)
    best_a = 1
    best_b = TARGET - 1
    
    # We only need to check a up to TARGET//2 because (a,b) and (b,a) are equivalent.
    # Actually, we just need to find the min.
    for a in range(1, TARGET // 2 + 1):
        b = TARGET - a
        pa = lpf[a]
        pb = lpf[b]
        current_max = pa if pa > pb else pb
        
        if current_max < best_max_p:
            best_max_p = current_max
            best_a = a
            best_b = b
            
            # Early exit: If we find a pair where max(P(a), P(b)) is 2,
            # that's the absolute minimum possible for integers > 1.
            # P(1)=1, but 1 is not prime. If one number is 1, P(1)=1.
            # If we find a pair involving 1, max(1, P(TARGET-1)).
            # TARGET-1 = 123456.
            # Let's just let the loop run, it's fast.
            
            # Can we break early?
            # The minimum possible value for max(P(a), P(b)) is 1 (if a=1, b=1, sum=2, not our case).
            # For sum > 2, at least one number is > 1.
            # If we find max = 2, it means both are powers of 2.
            # 2^x + 2^y = 123457 (odd). One must be odd power of 2? No, powers of 2 are even (except 2^0=1).
            # 1 + 123456. P(1)=1, P(123456)=?
            # 123456 is divisible by 2. 123456 / 2 = 61728. ...
            # It's unlikely to be a power of 2.
            # The minimum possible non-1 prime factor is 2.
            # If we find a pair with max factor 2, we can't do better than 2 (unless 1 is involved, but 1+123456 has P(123456) > 2 likely).
            # Actually, if we find max=2, it's the global minimum for pairs where both > 1.
            # If we find a pair involving 1, max is P(TARGET-1).
            # Let's just finish the loop. It's < 1ms.
            
    # Check time just in case, though it should be instant.
    if time.time() - start_time > DEADLINE:
        # Fallback
        return [1, TARGET - 1]
        
    return [best_a, best_b]
