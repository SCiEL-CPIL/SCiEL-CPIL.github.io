import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 115  # safety margin before 120s limit

    # Deterministic Miller-Rabin for n < 3.3 * 10^24
    def is_prime(n):
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in small_primes:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # Pollard's Rho with Brent's improvement
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            x = random.randint(2, n - 1)
            y = x
            d = 1
            c = random.randint(1, n - 1)
            m = 128
            g = q = r = 1
            while d == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                k = 0
                while k < r and d == 1:
                    ys = y
                    for _ in range(min(m, r - k)):
                        y = (y * y + c) % n
                        q = q * abs(x - y) % n
                    g = math.gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                d = 1
                while d == 1:
                    ys = (ys * ys + c) % n
                    d = math.gcd(abs(x - ys), n)
            if g != n:
                return g

    # Trial division by small primes first, then Pollard-Rho
    def factorize(n, small_primes):
        """Return list of prime factors of n (with multiplicity)."""
        factors = []
        for p in small_primes:
            if p * p > n:
                break
            while n % p == 0:
                factors.append(p)
                n //= p
        if n > 1:
            if is_prime(n):
                factors.append(n)
            else:
                # Use Pollard-Rho recursively
                stack = [n]
                while stack:
                    m = stack.pop()
                    if m == 1:
                        continue
                    if is_prime(m):
                        factors.append(m)
                    else:
                        d = pollard_rho(m)
                        stack.append(d)
                        stack.append(m // d)
        return factors

    def max_prime_factor(n, small_primes):
        """Return the largest prime factor of n."""
        max_pf = 1
        for p in small_primes:
            if p * p > n:
                break
            while n % p == 0:
                max_pf = max(max_pf, p)
                n //= p
        if n > 1:
            if is_prime(n):
                max_pf = max(max_pf, n)
            else:
                factors = factorize(n, small_primes)
                max_pf = max(max_pf, max(factors))
        return max_pf

    # Sieve primes up to limit
    def sieve(limit):
        if limit < 2:
            return []
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                step = i
                start = i * i
                is_p[start:limit + 1:step] = bytearray(len(is_p[start:limit + 1:step]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    # Generate primes up to p using segmented sieve for large p
    def generate_primes_up_to(p):
        if p < 2:
            return []
        if p < 100000:
            return sieve(p)
        # Use segmented sieve for larger p
        primes_small = sieve(10000)
        limit = int(math.isqrt(p))
        primes_for_sieve = [q for q in primes_small if q <= limit]
        
        low = 10000
        high = p
        size = high - low + 1
        is_prime_seg = bytearray(b'\x01') * size
        
        for q in primes_for_sieve:
            start = max(q * q, ((low + q - 1) // q) * q)
            for j in range(start - low, size, q):
                is_prime_seg[j] = 0
        
        result = primes_small[:]
        for i in range(low, high + 1):
            if is_prime_seg[i - low]:
                result.append(i)
        return result

    # For a given prime p, check the run length k
    # Every integer p^2+1, ..., p^2+k must have all prime factors at most p
    def check_run_length(p, max_k=200, primes_le_p=None):
        p2 = p * p
        
        if primes_le_p is None:
            if p <= 100000:
                primes_le_p = sieve(p)
            else:
                primes_le_p = generate_primes_up_to(p)
        
        # For each number p^2 + i, track its remaining factor after dividing by all primes <= p
        rem = [1] * (max_k + 1)
        for i in range(1, max_k + 1):
            rem[i] = p2 + i
        
        # For each prime q <= p, find multiples in [p^2+1, p^2+max_k]
        for q in primes_le_p:
            r = (-p2) % q
            if r == 0:
                r = q
            
            i = r
            while i <= max_k:
                while rem[i] % q == 0:
                    rem[i] //= q
                i += q
        
        # Find the longest run from i=1 where rem[i] == 1
        k = 0
        for i in range(1, max_k + 1):
            if rem[i] == 1:
                k += 1
            else:
                break
        
        return k

    best_k = 0
    best_p = 2

    # Precompute small primes for trial division
    small_primes = sieve(10000)
    
    # Initial small check
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        k = check_run_length(p, 200, small_primes)
        if k > best_k:
            best_k = k
            best_p = p

    # Search strategy:
    # 1. First, quickly scan primes up to ~50000 using the efficient method
    # 2. Then, for larger primes, use a smarter approach:
    #    - Precompute which primes q <= p can divide p^2+i for i in [1,200]
    #    - For each candidate p, only check the relevant primes
    
    # Phase 1: Scan primes up to 100000
    if time.time() - start_time < deadline - 30:
        primes_100k = sieve(100000)
        for p in primes_100k:
            if p <= 47:
                continue
            if time.time() - start_time > deadline - 15:
                break
            k = check_run_length(p, 200, [q for q in primes_100k if q <= p])
            if k > best_k:
                best_k = k
                best_p = p

    # Phase 2: For larger primes, use a different strategy
    # Instead of generating all primes up to p, we can be smarter:
    # For p^2 + i to be p-smooth, all prime factors must be <= p
    # Key insight: if q > p is a prime factor of p^2+i, then q > p
    # But q divides p^2+i, so q <= p^2+i
    # We need to verify that no prime factor exceeds p
    
    # Alternative approach for larger p:
    # Use the fact that for p^2+i to be p-smooth, we need to check
    # that after dividing by all primes <= p, the remainder is 1
    # But we can optimize by noting that if q > sqrt(p^2+i), then q is prime
    # and we just need to check if q <= p
    
    # Let's try a hybrid approach:
    # For each candidate p, factor p^2+i using trial division up to min(p, sqrt(p^2+i))
    # Since sqrt(p^2+i) ≈ p, trial division up to p is the same as before
    
    # Better idea: Use the structure more
    # p^2 + i = (p + a)(p - a + ... ) doesn't work simply
    # But p^2 + i for small i has limited factorization patterns
    
    # Let's try scanning primes in ranges and using a fast check
    # The key optimization: for large p, most p^2+i will have a prime factor > p
    # unless they are specially structured
    
    # Phase 3: Scan primes from 100000 to 10^6 in batches
    if time.time() - start_time < deadline - 10:
        # Generate primes up to 10^6
        primes_1m = sieve(1000000)
        
        # For each prime p in (100000, 10^6], check run length
        # Optimization: precompute for each prime q <= 10^6, 
        # the residues r_q = -p^2 mod q for various p
        # But this depends on p, so we can't fully precompute
        
        # Instead, let's just iterate and check
        for p in primes_1m:
            if p <= 100000:
                continue
            if time.time() - start_time > deadline - 5:
                break
            
            # Quick filter: check if p^2+1 has all factors <= p
            # If not, skip this p entirely
            n = p * p + 1
            rem = n
            for q in small_primes:
                if q > p:
                    break
                while rem % q == 0:
                    rem //= q
            if rem > 1:
                # Check if rem is prime and <= p
                if is_prime(rem):
                    if rem > p:
                        continue
                else:
                    # rem has a factor > p potentially
                    # Quick check: if rem > p*p, it likely has a large factor
                    # But we need to be careful
                    # For now, just continue and check more carefully
                    pass
            
            k = check_run_length(p, 200, [q for q in primes_1m if q <= p])
            if k > best_k:
                best_k = k
                best_p = p

    # Phase 4: Try some larger primes if time permits
    if time.time() - start_time < deadline - 5:
        # Try primes around 10^6 to 5*10^6
        # Generate primes in [10^6, 5*10^6]
        low = 1000000
        high = 5000000
        size = high - low + 1
        base_primes = [q for q in small_primes if q <= int(math.isqrt(high))]
        
        is_prime_seg = bytearray(b'\x01') * size
        for q in base_primes:
            start = max(q * q, ((low + q - 1) // q) * q)
            for j in range(start - low, size, q):
                is_prime_seg[j] = 0
        
        # Check candidates
        for offset in range(2, size, 2):  # Skip even numbers
            if is_prime_seg[offset]:
                p = low + offset
                if time.time() - start_time > deadline - 2:
                    break
                
                # Quick check on p^2+1
                n = p * p + 1
                rem = n
                for q in small_primes:
                    if q > p:
                        break
                    while rem % q == 0:
                        rem //= q
                if rem > 1 and rem > p:
                    continue
                
                # Full check - but this requires primes up to p
                # For p up to 5*10^6, generating primes is expensive
                # Let's use a different approach: factor each p^2+i directly
                k = 0
                for i in range(1, 201):
                    n = p * p + i
                    # Factor n and check all prime factors <= p
                    rem = n
                    valid = True
                    for q in small_primes:
                        if q > p:
                            break
                        if q * q > rem:
                            break
                        while rem % q == 0:
                            rem //= q
                    if rem > 1:
                        if is_prime(rem):
                            if rem > p:
                                valid = False
                        else:
                            # rem is composite, check its factors
                            factors = factorize(rem, small_primes)
                            if any(f > p for f in factors):
                                valid = False
                    if not valid:
                        break
                    k += 1
                
                if k > best_k:
                    best_k = k
                    best_p = p

    return {'p': best_p, 'k': best_k}

if __name__ == "__main__":
    res = construct()
    print(res)
