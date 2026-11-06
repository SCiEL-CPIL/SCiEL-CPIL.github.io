"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def now():
        return time.time()

    # Precompute primes up to 10^6 for trial division of numbers up to 10^12
    LIMIT = 10**6
    is_prime = bytearray(b'\x01') * (LIMIT + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, LIMIT + 1) if is_prime[i]]

    def is_prime_big(n):
        """Miller-Rabin primality test for large n."""
        if n < 2:
            return False
        if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            return True
        if n % 2 == 0 or n % 3 == 0 or n % 5 == 0 or n % 7 == 0:
            return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    def factor(n):
        """Factor n using trial division by small primes, then Pollard's rho for remainder."""
        factors = []
        for p in primes:
            if p * p > n:
                break
            while n % p == 0:
                factors.append(p)
                n //= p
        if n > 1:
            if is_prime_big(n):
                factors.append(n)
            else:
                # Pollard's rho for composite remainder
                def pollard_rho(m):
                    if m % 2 == 0:
                        return 2
                    x, y, d = 2, 2, 1
                    while d == 1:
                        x = (x * x + 1) % m
                        y = (y * y + 1) % m
                        y = (y * y + 1) % m
                        d = math.gcd(abs(x - y), m)
                    if d == m:
                        return pollard_rho(m)
                    return d
                def factor_recursive(m):
                    if m == 1:
                        return
                    if is_prime_big(m):
                        factors.append(m)
                        return
                    d = pollard_rho(m)
                    factor_recursive(d)
                    factor_recursive(m // d)
                factor_recursive(n)
        return factors

    def phi(x):
        """Compute Euler's totient function using factorization."""
        if x == 1:
            return 1
        factors = factor(x)
        result = x
        for p in set(factors):
            result -= result // p
        return result

    # Search for the longest run of consecutive integers with distinct totients
    # We search in windows and track the best run found so far
    best_k = 1
    best_s = 1

    # First, do a quick scan of small numbers to find a good starting point
    # The seed program found runs up to length ~14 in [1, 20000]
    # Let's verify and then extend the search to larger ranges

    # Phase 1: Scan [1, 50000] with precomputed phi values
    max_scan = 50000
    phi_vals = [0] * (max_scan + 1)
    for i in range(1, max_scan + 1):
        phi_vals[i] = phi(i)

    for s in range(1, max_scan):
        seen = set()
        k = 0
        while s + k <= max_scan and phi_vals[s + k] not in seen:
            seen.add(phi_vals[s + k])
            k += 1
        if k > best_k:
            best_k = k
            best_s = s
        if now() > DEADLINE:
            break

    # Phase 2: If we have time, try larger ranges with random sampling
    # Look for runs starting at various points up to 10^12
    if now() < DEADLINE - 5:
        # Try some strategically chosen starting points
        # Runs of distinct totients tend to occur where numbers have varied factorizations
        # Let's try ranges around multiples of highly composite numbers
        test_points = [
            10**6, 2*10**6, 5*10**6, 10**7, 2*10**7, 5*10**7, 10**8,
            2*10**8, 5*10**8, 10**9, 2*10**9, 5*10**9, 10**10, 2*10**10,
            5*10**10, 10**11, 2*10**11, 5*10**11, 9*10**11
        ]
        
        for tp in test_points:
            if now() > DEADLINE - 5:
                break
            # Try a window of 200 around this point
            s_start = max(1, tp - 100)
            s_end = min(10**12 - 199, tp + 100)
            
            # Compute phi for this window
            window_phi = {}
            for x in range(s_start, s_end + 1):
                if x > 10**12:
                    break
                window_phi[x] = phi(x)
            
            # Find best run in this window
            for s in range(s_start, s_end + 1):
                if s + 200 > 10**12 + 1:
                    break
                seen = set()
                k = 0
                while s + k <= s_end and s + k <= 10**12:
                    if window_phi[s + k] not in seen:
                        seen.add(window_phi[s + k])
                        k += 1
                    else:
                        break
                if k > best_k:
                    best_k = k
                    best_s = s
    
    # Ensure the result is valid
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k < 1:
        best_k = 1
        best_s = 1

    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
