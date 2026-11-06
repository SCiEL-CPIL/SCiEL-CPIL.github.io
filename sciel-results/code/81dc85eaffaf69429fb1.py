import math
import time

# Deterministic Miller-Rabin for n < 2^64
def is_prime(n):
    if n < 2:
        return False
    # Small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n % p == 0:
            return n == p
    # Write n-1 as d * 2^r
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    # Bases sufficient for n < 3,317,044,064,679,887,385,961,981 (well above 10^18)
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    y, c, m = 2, 1, 128
    g, r, q = 1, 1, 1
    while g == 1:
        x = y
        for _ in range(r):
            y = (y * y + c) % n
        k = 0
        while k < r and g == 1:
            ys = y
            for _ in range(min(m, r - k)):
                y = (y * y + c) % n
                q = q * abs(x - y) % n
            g = math.gcd(q, n)
            k += m
        r *= 2
    if g == n:
        while True:
            ys = (ys * ys + c) % n
            g = math.gcd(abs(x - ys), n)
            if g > 1:
                break
    return g

def largest_prime_factor(n, p_limit):
    """Return largest prime factor of n. If any factor > p_limit, return that factor (or larger)."""
    # First, handle small factors up to p_limit
    # We only care if largest factor <= p_limit
    # So we can trial divide up to min(sqrt(n), p_limit)
    # But we need to know if remaining part is <= p_limit
    
    # If n <= p_limit, check if prime
    if n <= p_limit:
        return n if is_prime(n) else n # Wait, if n is composite and <= p_limit, largest factor is <= n <= p_limit
        # Actually, if n <= p_limit, the largest factor is at most n, which is <= p_limit.
        # We just need to confirm it's not a factor > p_limit, which is impossible.
        # So return n (or any value <= p_limit).
    
    # Trial division by small primes up to min(sqrt(n), p_limit)
    # But we don't know sqrt(n) exactly if we stop at p_limit.
    # Let's optimize:
    # We want to know if n has any prime factor > p_limit.
    # If n has a factor > p_limit, it must be a prime (since we've removed all factors <= p_limit).
    # So we divide out all factors <= p_limit.
    # If remainder > 1, is it prime? And is it <= p_limit?
    # Wait, if we divide out all factors <= p_limit, the remainder R is either 1 or a prime > p_limit.
    # Because if R were composite, it would have a prime factor <= sqrt(R).
    # But we only divided up to p_limit.
    # This logic is tricky.
    
    # Better approach:
    # Factor n completely, but stop early if we find a factor > p_limit.
    # Or, simpler:
    # 1. Check small primes up to p_limit.
    # 2. If n becomes 1, largest factor was <= p_limit.
    # 3. If n > 1, is it prime?
    #    If n is prime, is n <= p_limit?
    #    If n is composite, it must have a factor <= sqrt(n).
    #    But we've already checked up to p_limit.
    #    If p_limit >= sqrt(n), then we've checked all possible small factors.
    #    So if n > 1 and p_limit >= sqrt(n), then n is prime.
    #    In that case, return n (and check if n <= p_limit).
    #    If p_limit < sqrt(n), then n might have a large prime factor.
    
    # Let's refine:
    # We want to return the largest prime factor.
    # If any prime factor > p_limit exists, the condition fails.
    # So we just need to check if ALL prime factors <= p_limit.
    
    m = n
    max_f = 1
    # Trial divide by 2
    while m % 2 == 0:
        m //= 2
        max_f = 2
    # Trial divide by odd numbers up to p_limit
    # But we should stop if d*d > m
    d = 3
    limit = min(p_limit, int(math.isqrt(m)) + 1)
    while d <= limit:
        if m % d == 0:
            max_f = d
            while m % d == 0:
                m //= d
            # Update limit if m decreased
            if m == 1:
                return max_f
            # Recompute sqrt bound? 
            # The loop condition d <= limit uses old limit. 
            # If m became smaller, we might stop earlier.
            # But limit was min(p_limit, old_sqrt). 
            # If m is now small, d might be > sqrt(m).
            # Let's check if m is 1.
            # If m > 1, and d > sqrt(m), then m is prime.
            # But we are in a while loop.
            # Let's break out and handle remainder.
            # Actually, updating limit dynamically is complex.
            # Let's just use Pollard-Brent for the remainder if it's large.
            pass
        d += 2
    
    # This naive loop is too slow if p_limit is large (up to 10^9).
    # We need a better strategy.
    
    # Strategy:
    # 1. Trial divide by small primes (e.g., up to 10^5 or 10^6).
    # 2. If remainder > 1:
    #    a. If remainder <= p_limit: Check if prime. If yes, max_f = max(max_f, remainder). If no, factor it (but it's small).
    #    b. If remainder > p_limit:
    #       - If remainder is prime: return remainder (which is > p_limit, so condition fails).
    #       - If remainder is composite: it must have a factor <= sqrt(remainder).
    #         If that factor <= p_limit, we could have found it in step 1? No, step 1 only went to 10^5/10^6.
    #         So we need to factor remainder.
    #         But if remainder has a factor > p_limit, we fail.
    #         So we need to check if remainder has any prime factor > p_limit.
    #         This is equivalent to: divide remainder by all primes <= p_limit. If remainder becomes 1, success. Else, fail.
    
    # Revised Strategy:
    # 1. Trial divide by primes up to B (e.g., 10^5).
    # 2. Let m be the remainder.
    # 3. If m == 1: return max_f.
    # 4. If m <= p_limit:
    #    - If is_prime(m): return max(max_f, m) (and check <= p_limit).
    #    - Else: Factor m (it's small) and update max_f.
    # 5. If m > p_limit:
    #    - We need to check if m has any prime factor > p_limit.
    #    - This means we need to remove all prime factors <= p_limit from m.
    #    - But we can't trial divide up to p_limit (10^9).
    #    - However, if m has a prime factor q > p_limit, then q is a prime factor of n.
    #    - We can use Pollard-Brent to find a factor of m.
    #    - If factor d is found:
    #       - If d > p_limit: fail (return d).
    #       - If d <= p_limit: divide m by d, recurse/check d.
    #    - Repeat until m == 1 or we find a factor > p_limit.
    #    - But Pollard-Brent finds *a* factor, not necessarily the smallest.
    #    - If it finds a large factor > p_limit, we can immediately return it.
    #    - If it finds a small factor <= p_limit, we divide it out and continue.
    #    - This might be slow if there are many small factors and a large one, but usually Pollard-Brent is fast.
    
    # Let's implement this recursive/iterative check.
    
    # Pre-check small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    max_f = 1
    for p in small_primes:
        while n % p == 0:
            n //= p
            max_f = max(max_f, p)
    if n == 1:
        return max_f
    if n > p_limit:
        # We need to verify if n has any factor > p_limit
        # Actually, if n > p_limit, it's not enough to just check primality.
        # If n is prime and n > p_limit, return n.
        # If n is composite, it has factors.
        
        # Let's use a helper to check if all factors of n are <= p_limit
        def all_factors_le(x, limit):
            if x == 1:
                return True
            if x <= limit:
                # If x is prime, its largest factor is x <= limit.
                # If x is composite, its largest factor is < x <= limit.
                return True
            # x > limit
            # Find a factor of x
            if is_prime(x):
                return False # x is a prime factor > limit
            d = pollard_brent(x)
            # d is a factor of x
            if d > limit:
                return False
            # d <= limit
            # Check if d's factors are <= limit (they are, since d <= limit)
            # Check if x/d's factors are <= limit
            return all_factors_le(x // d, limit)
        
        # This recursion might be deep or slow.
        # Let's do it iteratively.
        m = n
        while m > 1:
            if m <= p_limit:
                # All remaining factors are <= m <= p_limit
                break
            if is_prime(m):
                return m # m > p_limit, so this is a factor > p_limit
            d = pollard_brent(m)
            if d > p_limit:
                return d
            # d <= p_limit
            while m % d == 0:
                m //= d
            # Continue loop with reduced m
        return max_f # All factors were <= p_limit
    
    # Case: n <= p_limit but we haven't fully factored it?
    # We divided out small primes.
    # If n <= p_limit, we can just return n if is_prime(n) else ...
    # Actually, if n <= p_limit, the largest factor is <= n <= p_limit.
    # So we can just return n (or max_f if n==1, but we handled n==1).
    # Wait, if n is composite and <= p_limit, its largest factor is <= n <= p_limit.
    # So we don't need to factor it further to check the bound.
    # We just need to return a value <= p_limit.
    return max(max_f, n) if n > 1 else max_f

def largest_prime_factor_optimized(n, p_limit):
    """
    Returns the largest prime factor of n.
    If the largest prime factor is > p_limit, it returns a factor > p_limit (or the largest).
    If all prime factors are <= p_limit, it returns the largest one (<= p_limit).
    """
    if n <= 1:
        return 1
    
    # 1. Trial division by small primes
    # We can go up to a certain threshold to speed up Pollard-Brent
    # But for simplicity, let's just use Pollard-Brent with a quick check.
    
    # If n is prime
    if is_prime(n):
        return n
    
    m = n
    max_f = 1
    
    # Handle 2
    while m % 2 == 0:
        m //= 2
        max_f = max(max_f, 2)
    if m == 1:
        return max_f
    
    # Handle 3
    while m % 3 == 0:
        m //= 3
        max_f = max(max_f, 3)
    if m == 1:
        return max_f
    
    # Now m is not divisible by 2 or 3.
    # We want to check if m has any factor > p_limit.
    # If m <= p_limit, we are safe (largest factor <= m <= p_limit).
    # But we need to update max_f if m is prime?
    # If m <= p_limit, largest factor of m is <= m.
    # So max_f = max(max_f, m) is safe? No, if m is composite, largest factor < m.
    # But we just need to return *a* value that is the largest factor.
    # Actually, the function returns the largest factor.
    # If m <= p_limit, we can just factor m completely?
    # Or just return m? No, m might be composite.
    # But if m <= p_limit, all its factors are <= m <= p_limit.
    # So the condition "all factors <= p_limit" is satisfied.
    # We just need to return the largest factor.
    # For m <= p_limit, we can just return m if is_prime(m), else ...
    # Actually, if we just need to know if the condition holds, we can return 1 (or any <= p_limit) if it holds.
    # But the task says "return the largest factor".
    # However, the caller only checks if `largest_factor <= p`.
    # So if we return a value that is the *true* largest factor, or just a value that correctly reflects whether the bound is exceeded, it might work.
    # Let's stick to returning the actual largest factor for correctness, but optimize the search.
    
    if m <= p_limit:
        # We can factor m using trial division or Pollard-Brent.
        # Since m <= 10^9 (p_limit), trial division up to sqrt(m) is ~31622 steps.
        # This is fast.
        max_f_m = m
        d = 5
        w = 2
        while d * d <= max_f_m:
            while max_f_m % d == 0:
                max_f_m //= d
                max_f = max(max_f, d)
            d += w
            w = 6 - w
        if max_f_m > 1:
            max_f = max(max_f, max_f_m)
        return max_f
    
    # m > p_limit
    # We need to check if m has a factor > p_limit.
    # If m is prime, return m (> p_limit).
    # If m is composite, find a factor d.
    # If d > p_limit, return d.
    # If d <= p_limit, divide m by d, and continue.
    
    while m > 1:
        if m <= p_limit:
            # Factor m completely (fast)
            max_f_m = m
            d = 5
            w = 2
            while d * d <= max_f_m:
                while max_f_m % d == 0:
                    max_f_m //= d
                    max_f = max(max_f, d)
                d += w
                w = 6 - w
            if max_f_m > 1:
                max_f = max(max_f, max_f_m)
            return max_f
        
        if is_prime(m):
            return m # m > p_limit
        
        d = pollard_brent(m)
        if d > p_limit:
            return d
        # d <= p_limit
        # Divide out all d
        while m % d == 0:
            m //= d
            max_f = max(max_f, d)
    
    return max_f

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s safety margin
    
    best_k = 0
    best_p = 2
    
    # Search for primes p
    # We can search up to some limit.
    # p^2 + k can be large.
    # Let's search p up to 10^6 first, then expand if time permits?
    # Or just search sequentially.
    
    def is_prime_simple(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        i = 3
        while i * i <= n:
            if n % i == 0: return False
            i += 2
        return True
    
    p = 3
    while time.time() < deadline:
        # Next prime
        while not is_prime_simple(p):
            p += 2
        
        # Check run for this p
        k = 0
        # We can early stop if time is running out
        while time.time() < deadline - 5: # leave 5s for next primes
            n = p * p + k + 1
            # Check if all factors of n are <= p
            # Use largest_prime_factor_optimized
            lf = largest_prime_factor_optimized(n, p)
            if lf <= p:
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
        
        # If k is 0, maybe skip small p?
        # p += 2
        # Next prime
        p += 2
        
        # Optimization: if p is small, k is likely small.
        # We can jump p?
        # No, just iterate.
        
        # If we found a large k, maybe we can stop?
        # No, keep searching for max.
        
    return {'p': best_p, 'k': best_k}

# To speed up prime generation, let's use a sieve for small p and Miller-Rabin for large p?
# But is_prime_simple is fine for p up to 10^6 (sqrt is 1000).
# For p up to 10^9, sqrt is 31622.
# is_prime_simple is O(sqrt(p)).
# If we search p up to 10^6, we do ~10^6 * 1000 = 10^9 ops? Too slow.
# We need a faster prime generator.

# Let's rewrite construct with a better prime generator.

def construct():
    start_time = time.time()
    deadline = 145.0
    
    best_k = 0
    best_p = 2
    
    # Generate primes up to some limit using Sieve of Eratosthenes
    # Then test larger primes with Miller-Rabin
    
    limit_sieve = 1000000
    is_prime_arr = [True] * (limit_sieve + 1)
    is_prime_arr[0] = is_prime_arr[1] = False
    for i in range(2, int(limit_sieve**0.5) + 1):
        if is_prime_arr[i]:
            for j in range(i*i, limit_sieve + 1, i):
                is_prime_arr[j] = False
    
    primes = [i for i, val in enumerate(is_prime_arr) if val]
    
    # Check primes from sieve
    for p in primes:
        if time.time() > deadline - 5:
            break
        k = 0
        while time.time() < deadline - 5:
            n = p * p + k + 1
            lf = largest_prime_factor_optimized(n, p)
            if lf <= p:
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
    
    # Check larger primes
    # Start from limit_sieve + 1
    p = limit_sieve + 1
    if p % 2 == 0:
        p += 1
    
    while time.time() < deadline - 5:
        if is_prime(p):
            k = 0
            while time.time() < deadline - 5:
                n = p * p + k + 1
                lf = largest_prime_factor_optimized(n, p)
                if lf <= p:
                    k += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p
        p += 2
        
    return {'p': best_p, 'k': best_k}
