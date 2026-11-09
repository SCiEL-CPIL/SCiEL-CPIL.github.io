import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s buffer before 90s kill
    
    N_MIN = 10000
    N_MAX = 1000000
    MAX_VAL = N_MAX + 1  # we need n and n+1, so up to 1000001
    
    # powerful_part[i] will hold the powerful part of i
    powerful_part = [1] * (MAX_VAL + 1)
    
    # Sieve for smallest prime factor
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # For each prime p, for each power p^e with e >= 2,
    # multiply p^e into powerful_part[m] for all m divisible by p^e
    # We do this by iterating over primes and their powers
    
    # First, collect all primes
    primes = []
    for i in range(2, MAX_VAL + 1):
        if spf[i] == i:
            primes.append(i)
    
    # For each prime, for each power p^e (e>=2), multiply into multiples
    for p in primes:
        if p * p > MAX_VAL:
            break
        power = p * p
        while power <= MAX_VAL:
            # Multiply power into all multiples of power
            for m in range(power, MAX_VAL + 1, power):
                powerful_part[m] *= power
            power *= p
    
    # Now find n in [N_MIN, N_MAX] that maximizes powerful_part[n] * powerful_part[n+1]
    best_n = N_MIN
    best_val = powerful_part[N_MIN] * powerful_part[N_MIN + 1]
    
    for n in range(N_MIN, N_MAX + 1):
        val = powerful_part[n] * powerful_part[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    # Check deadline
    if time.time() > deadline - 2.0:
        pass  # We're already done with the scan
    
    return best_n
