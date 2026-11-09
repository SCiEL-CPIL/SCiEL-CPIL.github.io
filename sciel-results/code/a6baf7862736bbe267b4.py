import sys
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 1000000
    MAX_VAL = N + 1  # We need powerful_part for n and n+1, so up to 1000001
    
    # 1. Modified linear sieve to compute powerful part for all integers up to MAX_VAL
    # For each number, we track:
    # - lp[i]: largest prime factor of i
    # - pe[i]: p^e where p is the largest prime factor and e is its exact exponent in i
    # - powerful_part[i]: product of all prime powers p^e (e>=2) in i's factorization
    
    lp = [0] * (MAX_VAL + 1)  # largest prime factor
    pe = [1] * (MAX_VAL + 1)  # p^e for the largest prime factor
    powerful_part = [1] * (MAX_VAL + 1)
    primes = []
    
    for i in range(2, MAX_VAL + 1):
        if lp[i] == 0:
            # i is prime
            lp[i] = i
            pe[i] = i  # p^1, but we only care about e>=2
            primes.append(i)
        
        for p in primes:
            if p > lp[i] or i * p > MAX_VAL:
                break
            
            lp[i * p] = p
            
            if p == lp[i]:
                # p divides i, so the exponent of p in i*p is one more than in i
                if pe[i] == 1:
                    # p was not a factor of i (but now it is, as the largest prime)
                    # Actually, if p == lp[i], then p divides i
                    # pe[i] should be p^e for some e>=1
                    # But we initialized pe[i]=i for primes, which is wrong for this purpose
                    # Let me reconsider the approach
                    pass
                else:
                    pe[i * p] = pe[i] * p
            else:
                # p does not divide i, so p appears with exponent 1 in i*p
                pe[i * p] = p
            
            if time.time() > deadline - 2.0:
                break
        
        if time.time() > deadline - 1.0:
            break
    
    # The above approach is getting complicated. Let me use a simpler but still efficient method.
    # Use a sieve that for each prime power p^k (k>=2), marks all multiples.
    
    # Reset arrays
    powerful_part = [1] * (MAX_VAL + 1)
    
    # For each prime p, for each power p^k where k>=2, multiply powerful_part[m] by p^k
    # for all m that are divisible by p^k but not by p^(k+1)
    
    is_prime = [True] * (MAX_VAL + 1)
    is_prime[0] = False
    is_prime[1] = False
    primes = []
    
    for i in range(2, MAX_VAL + 1):
        if is_prime[i]:
            primes.append(i)
            if i * i <= MAX_VAL:
                for j in range(i * i, MAX_VAL + 1, i):
                    is_prime[j] = False
    
    for p in primes:
        power = p * p  # p^2
        while power <= MAX_VAL:
            next_power = power * p
            
            if next_power > MAX_VAL:
                # All multiples of power have exponent exactly k
                for m in range(power, MAX_VAL + 1, power):
                    powerful_part[m] *= power
            else:
                for m in range(power, MAX_VAL + 1, power):
                    if m % next_power != 0:
                        powerful_part[m] *= power
            
            power = next_power
            
            if time.time() > deadline - 3.0:
                break
    
    # 2. Find n in [10000, 1000000] maximizing pow_part[n] * pow_part[n+1]
    max_val = 0
    best_n = 10000
    
    for n in range(10000, N + 1):
        val = powerful_part[n] * powerful_part[n + 1]
        if val > max_val:
            max_val = val
            best_n = n
        
        if time.time() > deadline:
            break
            
    return best_n
