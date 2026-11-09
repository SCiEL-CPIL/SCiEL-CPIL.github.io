import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 1000000
    MAX_VAL = N + 1  # need n+1 up to 1000001
    
    # Build SPF table using a more efficient sieve
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL ** 0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # For each number, compute powerful part using SPF
    # powerful_part[n] = product of p^e where e>=2 in factorization of n
    pow_part = [1] * (MAX_VAL + 1)
    
    # Precompute prime powers for faster lookup
    # For each number, we'll track the powerful part incrementally
    # Use a sieve-based approach: for each prime p, for each power p^k (k>=2),
    # for each multiple m of p^k, we need to know if p^k is the FULL power in m
    # or if a higher power also divides m.
    
    # Alternative efficient approach:
    # pow_part[m] = product over primes p where v_p(m)>=2 of p^{v_p(m)}
    # We can compute this by: for each prime p, for each k>=2, for each m divisible by p^k
    # but not by p^{k+1}, multiply p^k into pow_part[m].
    
    # This is complex. Let's stick with SPF but optimize the inner loop.
    
    for i in range(2, MAX_VAL + 1):
        n = i
        pp = 1
        while n > 1:
            p = spf[n]
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            if e >= 2:
                pp *= p ** e
        pow_part[i] = pp
    
    # Now find n in [10000, 1000000] that maximizes pow_part[n] * pow_part[n+1]
    best_n = 10000
    best_val = pow_part[10000] * pow_part[10001]
    
    for n in range(10000, N + 1):
        val = pow_part[n] * pow_part[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    return best_n
