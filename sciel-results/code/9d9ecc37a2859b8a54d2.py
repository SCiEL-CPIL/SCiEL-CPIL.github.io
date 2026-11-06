import time
from math import isqrt

def _is_square_free_proper(n):
    """Proper square-free check using trial division up to cbrt(n)."""
    if n < 2:
        return True
    # Small primes for quick elimination
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        if p * p > n:
            break
        if n % (p * p) == 0:
            return False
        while n % p == 0:
            n //= p
    if n == 1:
        return True
    # Check if remaining n is a perfect square
    r = isqrt(n)
    if r * r == n:
        return False
    # Now n has no prime factor <= 97 with multiplicity >= 2
    # n is either 1, prime, product of two primes > 97, or p^k for p > 97
    # If n = p^2 for some prime p > 97, we caught it above
    # If n = p^3 for some prime p > 97, then p^3 > 97^3 = 912673
    # For n > 912673, we need to check if n is a perfect cube
    # But actually, for our purposes (b <= ~2*10^6), after removing primes <= 97,
    # the remaining n is small enough. Let's do a more careful check.
    # Trial divide by odd numbers up to cbrt(n)
    i = 99
    while i * i * i <= n:
        if n % (i * i) == 0:
            return False
        while n % i == 0:
            n //= i
        i += 2
    # Final check: is n a perfect square?
    r = isqrt(n)
    if r * r == n:
        return False
    return True


def _generate_powerful(limit):
    """Generate all powerful numbers <= limit.
    Powerful numbers are of the form a^2 * b^3 where b is square-free.
    """
    powerful = set([1])
    
    # Max b: b^3 <= limit, so b <= limit^(1/3)
    max_b = int(limit ** (1/3)) + 2
    
    # For each square-free b, a^2 * b^3 <= limit => a <= sqrt(limit / b^3)
    for b in range(1, max_b + 1):
        if not _is_square_free_proper(b):
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    
    return sorted(powerful)


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer before 120s limit
    
    # Strategy: Construct N as a multiple of a highly composite square-free integer Q
    # and search for representations where a = x^2 * Q and b = y^2 * Q
    # This reduces the problem to finding x^2 + y^2 = N/Q
    
    # Generate powerful numbers up to a reasonable limit for verification
    limit_pw = 10**11
    powerful = _generate_powerful(limit_pw)
    powerful_set = set(powerful)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time - 30:
        limit_pw = 10**9
        powerful = _generate_powerful(limit_pw)
        powerful_set = set(powerful)
    
    # Use the primorial approach
    # Q = product of first k primes
    # N = Q * M where M is a sum of two squares
    # a = Q * x^2, b = Q * y^2 where x^2 + y^2 = M
    
    # First, let's try with a small primorial Q
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    best_n = None
    best_count = 0
    best_pairs = []
    
    # Try different primorials
    for k in range(2, min(8, len(primes))):
        if time.time() > deadline - 10:
            break
        
        Q = 1
        for i in range(k):
            Q *= primes[i]
        
        # N = Q * M where M = x^2 + y^2
        # We want N <= 10^18, so M <= 10^18 / Q
        
        max_M = 10**18 // Q
        
        # Find M with many representations as sum of two squares
        # M = x^2 + y^2 has many representations if M has many prime factors of form 4j+1
        
        # For simplicity, let's try M as a product of primes of form 4j+1
        # Primes of form 4j+1: 5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509, 521, 541, 557, 569, 577, 593, 601, 613, 617, 641, 653, 661, 673, 677, 701, 709, 713, 733, 757, 761, 769, 773, 797, 809, 821, 829, 853, 857, 863, 881, 883, 929, 937, 941, 953, 977, 997
        
        primes_4j1 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317, 337, 349, 353, 373, 389, 397, 401, 409, 421, 433, 449, 457, 461, 509, 521, 541, 557, 569, 577, 593, 601, 613, 617, 641, 653, 661, 673, 677, 701, 709, 713, 733, 757, 761, 769, 773, 797, 809, 821, 829, 853, 857, 863, 881, 883, 929, 937, 941, 953, 977, 997]
        
        # Try different combinations of primes_4j1
        for m in range(1, min(6, len(primes_4j1))):
            if time.time() > deadline - 5:
                break
            
            M = 1
            for i in range(m):
                M *= primes_4j1[i]
            
            if M > max_M:
                break
            
            N = Q * M
            
            if N > 10**18:
                break
            
            # Count representations of N as sum of two powerful numbers
            count = 0
            pairs = []
            for a in powerful:
                if 2 * a > N:
                    break
                b = N - a
                if b in powerful_set:
                    count += 1
                    pairs.append([a, b])
            
            if count > best_count:
                best_count = count
                best_n = N
                best_pairs = pairs
    
    # Also try the direct approach with random sampling
    import random
    n_pw = len(powerful)
    
    for _ in range(5000):
        if time.time() > deadline - 5:
            break
        
        idx1 = random.randint(0, n_pw - 1)
        idx2 = random.randint(0, n_pw - 1)
        a = powerful[idx1]
        b = powerful[idx2]
        n_candidate = a + b
        
        if n_candidate > 10**18:
            continue
        
        count = 0
        pairs = []
        for pw in powerful:
            if 2 * pw > n_candidate:
                break
            complement = n_candidate - pw
            if complement in powerful_set:
                count += 1
                pairs.append([pw, complement])
        
        if count > best_count:
            best_count = count
            best_n = n_candidate
            best_pairs = pairs
    
    if best_n is None:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_n, 'pairs': best_pairs}
