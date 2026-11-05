import time
from math import isqrt
from collections import Counter

def _is_powerful(n):
    """Check if n is a powerful number (every prime factor appears with exponent >= 2)."""
    if n < 1:
        return False
    if n == 1:
        return True
    temp = n
    # Check small primes
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if temp % (p * p) == 0:
            while temp % p == 0:
                temp //= p
        elif temp % p == 0:
            return False
    if temp > 1:
        # temp should be a perfect square (or 1)
        r = isqrt(temp)
        if r * r != temp:
            return False
    return True

def _generate_powerful_numbers(limit):
    """Generate all powerful numbers up to limit."""
    powerful = set()
    powerful.add(1)
    # n = a^2 * b^3 where b is square-free
    # b^3 <= limit, so b <= limit^(1/3)
    max_b = int(limit ** (1/3)) + 2
    
    # Pre-compute small primes for square-free check
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    for b in range(1, max_b + 1):
        # Check if b is square-free
        t = b
        is_sf = True
        for p in small_primes:
            if p * p > t:
                break
            if t % (p * p) == 0:
                is_sf = False
                break
            while t % p == 0:
                t //= p
        if not is_sf:
            continue
        if t > 1:
            # t is a prime > 97 (or product of such, but b is small so t is prime)
            # If t > 1, it appears to power 1, so b is square-free.
            # Wait, if b has a prime factor > 97, it must appear to power 1 for b to be square-free?
            # Yes, if it appeared to power 2, b would be divisible by p^2, and since p>97, p^2 > 9409.
            # max_b is ~2154 for 10^10. So b < 2154.
            # Primes > 97 up to 2154.
            # If t > 1, it is a prime factor of b that was not removed.
            # Since we checked all p <= 97, t must be a prime > 97.
            # It appears to power 1 in b (since if it appeared to power 2, b >= p^2 > 97^2 = 9409, 
            # but we only check p up to 97. Wait.
            # If b = 101^2 = 10201. max_b is 2154. So b cannot be 10201.
            # So any prime factor > 97 in b must appear to power 1.
            # So t > 1 implies b is square-free.
            pass
        
        b3 = b ** 3
        if b3 > limit:
            continue
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    # Generate powerful numbers up to 10^10
    # K approx 5669
    pow_limit = 10**10
    powerful = _generate_powerful_numbers(pow_limit)
    K = len(powerful)
    
    # Check time
    if time.time() > deadline - 60:
        # If generation took too long, reduce limit? 
        # But generation of 10^10 is fast (< 1s).
        pass

    # Count sums
    # K ~ 5669, K^2 ~ 32 million
    # This should take ~ 5-10 seconds in Python
    sum_counts = Counter()
    
    # Optimization: Use local variables for speed
    pow_list = powerful
    counts = sum_counts
    
    for i in range(K):
        a = pow_list[i]
        # b must be >= a, so j >= i
        for j in range(i, K):
            b = pow_list[j]
            s = a + b
            counts[s] += 1
            if time.time() > deadline - 30:
                # If we are running out of time, stop counting
                # We will use whatever counts we have
                break
        if time.time() > deadline - 30:
            break
    
    if not counts:
        return {'N': 5, 'pairs': [[1, 4]]}
        
    best_N = max(counts, key=counts.get)
    best_count = counts[best_N]
    
    # Extract pairs
    pow_set = set(pow_list)
    pairs = []
    for a in pow_list:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in pow_set and a <= b:
            pairs.append([a, b])
            
    # Verify
    verified_pairs = []
    for p in pairs:
        if _is_powerful(p[0]) and _is_powerful(p[1]):
            verified_pairs.append(p)
            
    if not verified_pairs:
        verified_pairs = [[1, 4]]
        best_N = 5
        
    return {'N': best_N, 'pairs': verified_pairs}
