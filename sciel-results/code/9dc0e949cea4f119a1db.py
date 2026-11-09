import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s buffer before 90s kill
    
    N = 1000000
    MAX_N = N + 1  # we need n and n+1, so up to 1000001
    
    # powerful_part[i] will hold the powerful part of i
    powerful_part = [1] * (MAX_N + 1)
    
    # Sieve for smallest prime factor
    spf = list(range(MAX_N + 1))
    for i in range(2, int(MAX_N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # For each prime p, for each power p^e with e >= 2,
    # multiply p^e into powerful_part[m] for all m divisible by p^e
    # We do this by iterating over primes and their powers
    
    # First, collect all primes
    primes = []
    for i in range(2, MAX_N + 1):
        if spf[i] == i:
            primes.append(i)
    
    # For each prime, iterate over powers
    for p in primes:
        power = p * p  # p^2
        while power <= MAX_N:
            # Multiply power into powerful_part[m] for all m that are multiples of power
            # But we need to be careful: if m is divisible by p^e for e >= 2,
            # we want to include p^e in the powerful part.
            # However, if m is divisible by p^3, we want p^3 (not p^2 * p^1).
            # 
            # Actually, the powerful part is the product of p^e where e is the FULL exponent.
            # So if m = p^3 * q, the powerful part includes p^3 (not p^2).
            #
            # The sieve approach: for each p^e (e>=2), we should multiply p^e into
            # powerful_part[m] for m divisible by p^e BUT NOT by p^(e+1).
            # Wait, that's not right either.
            #
            # Let me reconsider. The powerful part of n is the product over all primes p
            # of p^(e_p) where e_p >= 2, and e_p is the exponent of p in n.
            #
            # So if n has p^3, the powerful part includes p^3.
            # If n has p^2, the powerful part includes p^2.
            #
            # The correct sieve: for each prime p and exponent e >= 2,
            # for each m divisible by p^e but NOT by p^(e+1), multiply powerful_part[m] by p^e.
            #
            # But that's complex. Alternative: for each m, factorize it and compute powerful_part directly.
            # With SPF, factorization is O(log n) per number, so total O(N log N) which is fine.
            
            break  # Skip the sieve approach, use direct factorization
    
    # Direct approach: for each n, compute powerful_part(n) using SPF
    # This is O(N * log N) which should be fast enough
    
    # Actually, let me use a more efficient sieve:
    # For each prime p, for each power p^e (e >= 2):
    #   For each multiple m of p^e:
    #     If m is NOT divisible by p^(e+1), then powerful_part[m] *= p^e
    #   Else (m is divisible by p^(e+1)), we'll handle it at the next power level
    
    # Even simpler: for each prime p, for each power p^e (e >= 2):
    #   For each multiple m of p^e:
    #     powerful_part[m] *= p  (just multiply by p once)
    # This way, if m is divisible by p^e, we multiply by p for each e from 2 to actual_exponent.
    # So if actual exponent is 3, we multiply by p for e=2 and e=3, giving p^2. But we want p^3!
    
    # Correct approach: if m is divisible by p^e where e is the full exponent,
    # we want to multiply by p^e. 
    # 
    # Sieve method: for each prime p, for each e >= 2:
    #   power = p^e
    #   For each multiple m of power:
    #     if m is not divisible by p^(e+1):
    #       powerful_part[m] *= power
    #
    # Let me implement this properly.
    
    powerful_part = [1] * (MAX_N + 1)
    
    for p in primes:
        power = p * p  # p^2
        e = 2
        while power <= MAX_N:
            # Check if p^(e+1) overflows
            next_power = power * p
            # For multiples of 'power' that are NOT multiples of 'next_power',
            # the exponent of p is exactly e
            for m in range(power, MAX_N + 1, power):
                if next_power > MAX_N or m % next_power != 0:
                    powerful_part[m] *= power
            power = next_power
            e += 1
    
    # Now find the best n in [10000, 1000000] maximizing powerful_part[n] * powerful_part[n+1]
    best_n = 10000
    best_val = powerful_part[10000] * powerful_part[10001]
    
    for n in range(10001, N + 1):
        if time.time() > deadline:
            break
        val = powerful_part[n] * powerful_part[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    return best_n
