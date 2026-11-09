import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s buffer before 90s kill
    
    N = 1000000
    MAX_VAL = N + 1  # We need n and n+1, so up to 1000001
    
    # 1. Sieve of Eratosthenes to find primes up to MAX_VAL
    is_prime = [True] * (MAX_VAL + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_VAL + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # 2. Compute powerful part for each number 1..MAX_VAL
    # powerful_part[i] = product of p^e for all prime factors p of i with exponent e >= 2
    pow_part = [1] * (MAX_VAL + 1)
    
    for p in primes:
        power = p * p  # p^2
        e = 2
        while power <= MAX_VAL:
            # For each multiple m of 'power', check if exponent of p is exactly e
            # m = k * power
            # m is divisible by p^(e+1) = power * p iff k is divisible by p
            # So we want k NOT divisible by p
            limit = MAX_VAL // power
            for k in range(1, limit + 1):
                if k % p != 0:
                    m = k * power
                    pow_part[m] *= power
            
            power *= p
            e += 1
            
            # Time check
            if time.time() > deadline:
                break
    
    # 3. Find n in [10000, 1000000] maximizing pow_part[n] * pow_part[n+1]
    max_val = 0
    best_n = 10000
    
    # Vectorized check might be faster, but list comprehension or loop is fine
    # Let's use a loop with early exit if time is critical, but this is O(N) so it's fast
    for n in range(10000, N + 1):
        val = pow_part[n] * pow_part[n + 1]
        if val > max_val:
            max_val = val
            best_n = n
        if time.time() > deadline:
            break
            
    return best_n
