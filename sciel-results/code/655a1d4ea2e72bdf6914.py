"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 130.0  # Safety margin before 150s limit
    
    # Precompute primes up to 10^6 for efficient factorization
    # For n <= 10^12, after removing all prime factors <= 10^6,
    # the remainder is either 1 or prime (since p*q > 10^12 if p,q > 10^6)
    limit_p = 10**6
    is_prime = bytearray(b'\x01') * (limit_p + 1)
    is_prime[0] = 0
    if limit_p >= 1:
        is_prime[1] = 0
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = 0
    
    # Extract primes list for trial division
    primes = [i for i in range(2, limit_p + 1) if is_prime[i]]
    
    def phi(x):
        """Compute Euler's totient function using trial division by primes."""
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result
    
    def check_run(s, k):
        """Check if phi(s), ..., phi(s+k-1) are pairwise distinct."""
        seen = set()
        for i in range(k):
            val = phi(s + i)
            if val in seen:
                return False
            seen.add(val)
        return True
    
    def find_max_run_from_start(s):
        """Find the maximum run length starting at s."""
        seen = set()
        k = 0
        while s + k <= 10**12:
            val = phi(s + k)
            if val in seen:
                break
            seen.add(val)
            k += 1
            if time.time() > DEADLINE - 5:
                break
        return k
    
    # Phase 1: Try small numbers (fast, likely short runs)
    best_k = 1
    best_s = 1
    
    # Check small range first
    small_limit = 10000
    values = [0] + [phi(x) for x in range(1, small_limit + 1)]
    for s in range(1, small_limit - 200):
        seen, k = set(), 0
        while s + k <= small_limit and values[s + k] not in seen:
            seen.add(values[s + k])
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
        if best_k >= 200:
            break
    
    if time.time() > DEADLINE - 10:
        return {'start': best_s, 'length': best_k}
    
    # Phase 2: Random sampling of large numbers
    # For large n, totient collisions are rare, so long runs are more likely
    random.seed(42)
    
    # Sample from different ranges
    num_attempts = 0
    while time.time() < DEADLINE - 10:
        num_attempts += 1
        
        # Choose a random starting point in a large range
        # Focus on ranges where n is large (10^8 to 10^12)
        if num_attempts % 3 == 0:
            s = random.randint(10**8, 10**12 - 200)
        elif num_attempts % 3 == 1:
            s = random.randint(10**6, 10**8)
        else:
            s = random.randint(10**4, 10**6)
        
        # Check if we can get a long run from this start
        # Use early termination for efficiency
        seen = set()
        k = 0
        for i in range(200):
            if time.time() > DEADLINE - 5:
                break
            val = phi(s + i)
            if val in seen:
                break
            seen.add(val)
            k += 1
        
        if k > best_k:
            best_k = k
            best_s = s
            if best_k >= 200:
                break
    
    # Phase 3: Local optimization around best start found
    if best_k < 200 and time.time() < DEADLINE - 15:
        # Try neighbors of the best start
        for offset in range(-50, 51):
            if time.time() > DEADLINE - 5:
                break
            s = best_s + offset
            if s < 1 or s + 200 > 10**12:
                continue
            seen = set()
            k = 0
            for i in range(200):
                val = phi(s + i)
                if val in seen:
                    break
                seen.add(val)
                k += 1
            if k > best_k:
                best_k = k
                best_s = s
                if best_k >= 200:
                    break
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
