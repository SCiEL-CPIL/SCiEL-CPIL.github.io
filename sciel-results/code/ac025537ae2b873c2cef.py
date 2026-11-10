import time
import math

def construct():
    start_time = time.time()
    deadline = 85  # seconds, safety margin
    
    MAX_N = 1000000
    MAX_VAL = MAX_N + 1  # we need to check up to 1000001
    
    # powerful_part[i] = product of p^e for all p^e || i with e >= 2
    powerful_part = [1] * (MAX_VAL + 1)
    
    # Sieve: for each prime p, for each power p^e with e>=2, mark multiples
    # For each number i, we want to extract the full p^e factor if e>=2
    
    # Better approach: for each prime p, compute p^2, p^3, ... and for each multiple,
    # determine the full power of p dividing that multiple
    
    # Initialize powerful_part as 1 for all
    # For each prime p, for each number n, find the highest e such that p^e | n
    # If e >= 2, multiply powerful_part[n] by p^e
    
    # Sieve for primes
    is_prime = [True] * (MAX_VAL + 1)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, MAX_VAL + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, MAX_VAL + 1, i):
                is_prime[j] = False
    
    # For each prime p, for each multiple of p^2, compute the full power
    for p in primes:
        p2 = p * p
        if p2 > MAX_VAL:
            break
        # For each multiple of p2, find the exact power of p
        # p^e divides n, find max e
        pe = p2
        while pe <= MAX_VAL:
            # All multiples of pe are divisible by p^2
            # For each multiple m of pe, find the highest power of p dividing m
            # Then if that power >= 2, multiply powerful_part[m] by p^(that power)
            # But we need to be careful: we should only multiply once per prime
            for m in range(pe, MAX_VAL + 1, pe):
                # Find the highest power of p dividing m
                temp = m
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count >= 2:
                    powerful_part[m] *= p ** count
            pe *= p
    
    # Now find the best n in [10000, MAX_N]
    best_product = 0
    best_n = 10000
    for n in range(10000, MAX_N + 1):
        prod = powerful_part[n] * powerful_part[n + 1]
        if prod > best_product:
            best_product = prod
            best_n = n
        if time.time() - start_time > deadline - 5:
            break
    
    return best_n
