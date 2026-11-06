import math
import time

def get_primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def generate_smooth_numbers(y, max_val):
    primes = get_primes_up_to(y)
    if not primes:
        return set()
    
    smooth = {1}
    current = [1]
    
    while current:
        next_current = []
        for u in current:
            for p in primes:
                v = u * p
                if v > max_val:
                    break
                if v not in smooth:
                    smooth.add(v)
                    next_current.append(v)
        current = next_current
        
    return smooth

def largest_prime_factor(n):
    if n <= 1:
        return 1
    max_p = 1
    d = 2
    while d * d <= n:
        while n % d == 0:
            max_p = d
            n //= d
        d += 1
    if n > 1:
        max_p = n
    return max_p

def construct():
    start_time = time.time()
    deadline = start_time + 50 # Safety margin
    
    # We want to find n in [10000, 1000000]
    min_n = 10000
    max_n = 1000000
    
    # Try y from small to large
    # Primes to try
    primes_to_try = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499]
    
    best_n = min_n # Fallback
    best_lpf = largest_prime_factor(min_n * (min_n + 1)) # Fallback
    
    for y in primes_to_try:
        if time.time() > deadline:
            break
            
        smooth_set = generate_smooth_numbers(y, max_n)
        
        # Find consecutive pairs
        candidates = []
        # Efficiently check consecutive
        # Convert to sorted list? Or just check s and s+1 in set
        # Since set lookup is O(1), we can just iterate over the set
        # But we only care about n in [min_n, max_n]
        for s in smooth_set:
            if s < min_n or s > max_n:
                continue
            # Check if s+1 is smooth
            if (s + 1) in smooth_set:
                # Pair (s, s+1)
                # Calculate exact LPF
                lpf_s = largest_prime_factor(s)
                lpf_sp1 = largest_prime_factor(s + 1)
                max_lpf = max(lpf_s, lpf_sp1)
                candidates.append((max_lpf, s))
        
        if candidates:
            # We found pairs with max prime factor <= y
            # Since we iterate y upwards, this is the minimal class
            # Find the best one
            candidates.sort()
            best_lpf, best_n = candidates[0]
            # We can stop because any future y will yield higher or equal LPF
            # Wait, could a future y yield a LOWER LPF?
            # No, if a pair has LPF < y, it would have been found in a smaller y iteration.
            # So the first non-empty candidates list contains the global optimum.
            break
            
    return best_n
