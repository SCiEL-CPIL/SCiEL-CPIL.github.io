import math
import time

def construct():
    # Sieve for primes up to a reasonable limit
    def get_primes(n):
        sieve = bytearray([True]) * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(200000)

    # Deterministic Miller-Rabin for 64-bit
    def is_prime(n):
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        bases = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
        for a in bases:
            if a % n == 0:
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

    # Check if n is p-smooth: all prime factors <= p
    # Key insight: for n = p^2 + i with i < p, n < (p+1)^2, so n has at most one prime factor > p
    # If n has a prime factor q > p, then q > p and cofactor r = n/q < (p^2+i)/p = p + i/p < p+1, so r <= p
    # Thus n = q * r with q > p, r <= p. So n is NOT p-smooth iff there exists r <= p with r | n and (n/r) > p and (n/r) is prime.
    # But simpler: divide out all small prime factors <= p. If remainder is 1, smooth. If remainder is prime <= p, smooth. Else not smooth.
    # But we can't divide by all primes <= p for large p. Instead:
    # n is p-smooth iff n has no prime factor > p.
    # We can check: does n have a prime factor in (p, n]? Since n < (p+1)^2, any prime factor > p must be > p and <= n.
    # But if q > p is a prime factor, then q > p and q <= n. The cofactor r = n/q <= n/(p+1) < p^2/p = p. So r < p.
    # So q = n/r for some r < p. We can iterate r over primes <= p and check if n % r == 0 and (n//r) > p and is_prime(n//r).
    # But that's still expensive.
    
    # Better: use the fact that we only need to check if n is p-smooth.
    # Trial divide by primes up to min(sqrt(n), p). After that, if remainder > 1, check if it's prime.
    # If remainder is prime and > p, not smooth. If remainder is prime and <= p, smooth. If remainder is 1, smooth.
    # If remainder is composite, it must have a factor <= sqrt(remainder) <= sqrt(n) <= p (since we divided by all primes up to sqrt(n) or p).
    # Wait, we divide by primes up to min(sqrt(n), p). If sqrt(n) <= p, we divide by all primes up to sqrt(n), remainder is 1 or prime.
    # If p < sqrt(n), we divide by primes up to p. Remainder could be composite with factors > p, but since we're checking p-smoothness, 
    # if remainder has a prime factor > p, it's not smooth. But remainder < n and all its prime factors > p (since we removed all <= p).
    # Since n < (p+1)^2, remainder < (p+1)^2 and all prime factors > p, so remainder can have at most one prime factor > p (since product of two > p is > p^2, and remainder < (p+1)^2 ≈ p^2).
    # Actually remainder could be a prime > p, or product of two primes both > p but their product < (p+1)^2.
    # If remainder is prime > p: not smooth.
    # If remainder is composite: it must be q1*q2 with q1,q2 > p and q1*q2 < (p+1)^2. So q1,q2 in (p, p+1) which is empty for integer primes. 
    # Wait, q1 > p and q2 > p implies q1*q2 > p^2. And remainder <= n < (p+1)^2 = p^2 + 2p + 1. So q1*q2 could be in (p^2, p^2+2p+1).
    # For example p=100, q1=101, q2=101, product=10201. p^2=10000, (p+1)^2=10201. So 101*101 = 10201 = (p+1)^2. But n < (p+1)^2, so n <= (p+1)^2 - 1 = 10200. So 101*101 > n. So no composite remainder with all factors > p is possible when n < (p+1)^2!
    
    # Therefore: after dividing by all primes <= p, the remainder is either 1 or a prime > p (not smooth) or a prime <= p (smooth, but we already divided it out).
    # Wait, we divide by primes <= p, so remainder has no prime factors <= p. If remainder > 1, it's either prime > p or composite with all factors > p. But we showed composite with all factors > p is impossible for n < (p+1)^2. So remainder is 1 or prime > p.
    
    # So: n is p-smooth iff after dividing by all primes <= p, remainder is 1.
    # But we can't iterate all primes <= p for large p. However, we only need to divide by primes up to sqrt(n) if sqrt(n) <= p, else up to p.
    # But if p < sqrt(n), we divide by primes up to p. Number of primes up to p is ~p/ln(p), which is too many for p ~ 10^9.
    
    # Alternative: just check if n has a prime factor > p by checking if n is prime (and > p), or if n has a factorization with a prime > p.
    # Since at most one prime factor > p, we can: for each prime r <= p, if r | n, check if (n/r) is prime and > p. But again too many r.
    
    # Practical approach: trial divide by primes up to a limit L. If after that, remainder is 1 or prime <= p, smooth. Else not.
    # Choose L = min(p, 10^6) or so. For p < 10^6, this is exact. For p > 10^6, we might miss a prime factor in (10^6, p], but that's unlikely to be the only large factor.
    
    # Actually, for the purpose of this task, let's just do: trial divide by primes up to min(p, 100000). Then if remainder > 1, check if prime. If prime and > p, not smooth. If prime and <= p, smooth. If composite, use Pollard-Rho to factor and check.
    
    def is_p_smooth(n, p, small_primes):
        if n <= 1:
            return True
        # Trial divide by small primes up to min(p, 100000)
        limit = min(p, 100000)
        for prime in small_primes:
            if prime > limit:
                break
            if n % prime == 0:
                while n % prime == 0:
                    n //= prime
        if n == 1:
            return True
        if n <= p:
            return True  # n is prime or 1, and <= p
        # n > p here. Check if n is prime
        if is_prime(n):
            return False  # prime > p
        # n is composite and > p. Factor it to find if any prime factor > p
        # Since n < (p+1)^2 for our use case, n has at most one prime factor > p
        # Try to find a factor
        # Use Pollard Rho
        import random
        def pollard_rho(m):
            if m % 2 == 0:
                return 2
            if m % 3 == 0:
                return 3
            while True:
                y = random.randrange(2, m)
                c = random.randrange(1, m)
                m_val = 128
                g = r = q = 1
                while g == 1:
                    x = y
                    for _ in range(r):
                        y = (y * y + c) % m
                    k = 0
                    while k < r and g == 1:
                        ys = y
                        for _ in range(min(m_val, r - k)):
                            y = (y * y + c) % m
                            q = (q * abs(x - y)) % m
                        g = math.gcd(q, m)
                        k += m_val
                    r *= 2
                if g == m:
                    while True:
                        ys = (ys * ys + c) % m
                        g = math.gcd(abs(x - ys), m)
                        if g > 1:
                            break
                if g != m:
                    return g
        
        d = pollard_rho(n)
        if is_p_smooth(d, p, small_primes) and is_p_smooth(n // d, p, small_primes):
            return True
        return False

    # Search for best (p, k)
    best_k = 0
    best_p = 2
    start_time = time.time()
    timeout = 115

    small_primes = [p for p in primes if p <= 100000]

    # Check primes up to a reasonable bound
    for p in primes:
        if p > 500000:
            break
        if time.time() - start_time > timeout:
            break
        k = 0
        for i in range(1, 201):
            n = p * p + i
            if not is_p_smooth(n, p, small_primes):
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                break

    return {'p': best_p, 'k': best_k}
