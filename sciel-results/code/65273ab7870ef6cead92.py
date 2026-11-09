import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N_MAX = 1000000
    N_MIN = 10000
    
    # Compute the powerful part for all integers up to N_MAX+1
    # powerful_part[n] = product of p^e for all prime factors p with e >= 2 in n's factorization
    
    # We use a sieve approach:
    # For each prime p, for each power p^2, p^3, ... up to N_MAX+1,
    # mark that this prime power contributes to the powerful part of its multiples.
    
    max_val = N_MAX + 1  # We need powerful_part up to N_MAX+1 (for n+1 when n = N_MAX)
    
    # powerful_part[i] will be the product of all prime powers p^e (e>=2) dividing i
    powerful_part = [1] * (max_val + 1)
    
    # Sieve for primes and their powers
    # For each prime p, for each power p^k (k>=2), multiply powerful_part[multiple] by p^k
    # if p^k divides multiple
    
    # First, find all primes up to max_val
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, max_val + 1) if is_prime[i]]
    
    # For each prime p, compute p^2, p^3, ... and mark multiples
    for p in primes:
        power = p * p
        while power <= max_val:
            # For each multiple of power, multiply powerful_part by p (for the additional exponent)
            # Wait, this is not quite right. Let me think again.
            # If p^k divides n, then p^k is part of the powerful part if k >= 2.
            # So for n divisible by p^k (k >= 2), we want to include p^k in powerful_part[n].
            # 
            # Approach: For each prime p, for each k >= 2, for each multiple of p^k,
            # we need to ensure that p^k (not p^(k-1) or other) is included.
            # Actually, the powerful part includes the FULL prime power p^e where e is the 
            # actual exponent in n's factorization.
            #
            # Better approach: For each prime p, compute the exponent of p in each number.
            # Then for numbers where exponent >= 2, multiply powerful_part by p^exponent.
            #
            # Even better: Use a modified sieve.
            # For each prime p, for each number n divisible by p^2, 
            # determine the exact exponent of p in n, and multiply powerful_part[n] by p^exponent.
            
            # Let's do it this way:
            # For each prime p, for each multiple m of p^2, 
            # count how many times p divides m, and multiply powerful_part[m] by p^count.
            
            # But this is O(N log N) which should be fine for N = 10^6.
            
            # Actually, let's use a cleaner approach:
            # For each prime p, iterate through multiples of p^2, p^3, etc.
            # For a number n, the exponent of p in n is the largest k such that p^k | n.
            # We can compute this by checking divisibility.
            
            # Let me restructure: for each prime p, for each multiple of p,
            # compute the exponent and update powerful_part.
            break  # We'll redo this below with a cleaner approach
    
    # Cleaner approach:
    # For each prime p, for each multiple of p, count the exponent of p.
    # If exponent >= 2, multiply powerful_part[multiple] by p^exponent.
    
    powerful_part = [1] * (max_val + 1)
    
    for p in primes:
        # For each multiple of p
        for m in range(p, max_val + 1, p):
            # Count exponent of p in m
            temp = m
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            if exp >= 2:
                powerful_part[m] *= (p ** exp)
    
    # Now scan for the best n in [N_MIN, N_MAX]
    best_val = 0
    best_n = N_MIN
    
    for n in range(N_MIN, N_MAX + 1):
        val = powerful_part[n] * powerful_part[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
    
    return best_n
