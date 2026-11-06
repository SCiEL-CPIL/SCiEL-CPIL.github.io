import math
import time

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    # Deterministic Miller-Rabin for n < 2^64
    def is_prime_mr(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Bases sufficient for n < 2^64
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    # Pollard-Brent factorization
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = 2
            c = 1
            m = 128
            g = 1
            r = 1
            q = 1
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
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
            if g != n:
                return g
            c += 1
    
    def factor(n, factors):
        if n == 1:
            return
        if is_prime_mr(n):
            factors.append(n)
            return
        d = pollard_brent(n)
        factor(d, factors)
        factor(n // d, factors)
    
    # Sieve of Eratosthenes
    def sieve(limit):
        is_prime_arr = [True] * (limit + 1)
        is_prime_arr[0] = is_prime_arr[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime_arr[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime_arr[j] = False
        return [i for i, prime in enumerate(is_prime_arr) if prime]
    
    # Small primes for trial division (up to ~10^6)
    small_primes = sieve(1000000)
    small_prime_set = set(small_primes)
    
    # Check if n is p-smooth
    def is_p_smooth(n, p, primes):
        if n <= 1:
            return True
        r = n
        # Trial division by primes <= min(p, sqrt(n))
        limit = min(p, int(n**0.5) + 1)
        for q in primes:
            if q > limit:
                break
            while r % q == 0:
                r //= q
        if r == 1:
            return True
        # r > 1: check if r is prime and <= p
        if r <= p and is_prime_mr(r):
            return True
        # r is composite or r > p: not smooth
        return False
    
    # Segmented sieve approach for rapid screening
    # For a given p, check p^2+1 through p^2+k
    # We can batch check using the primes up to p
    
    best_k = 0
    best_p = 2
    
    # Search through primes, starting with larger primes where longer runs are more likely
    # Generate primes up to some limit, then search in descending order or by heuristic
    # Actually, let's search all primes up to a limit and pick the best k
    
    # Strategy: 
    # 1. For small p (<= 10000), do full trial division - fast
    # 2. For larger p, use segmented approach
    
    # First, try small primes quickly
    for p in small_primes:
        if time.time() - start_time > deadline:
            break
        if p > 100000:
            break  # Small primes: just quick check
        
        p2 = p * p
        k = 0
        for offset in range(1, 201):
            if time.time() - start_time > deadline:
                break
            n = p2 + offset
            if is_p_smooth(n, p, small_primes):
                k = offset
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
    
    # For larger primes, we need a smarter approach
    # Generate primes in larger ranges and check them
    # Use segmented sieve to generate primes in [100000, 200000], etc.
    
    def sieve_segment(low, high, base_primes):
        """Generate primes in [low, high] using base primes."""
        size = high - low + 1
        is_prime_seg = [True] * size
        is_prime_seg[0] = (low == 0)
        if low == 1:
            is_prime_seg[0] = False
        for p in base_primes:
            if p * p > high:
                break
            # Find first multiple of p in [low, high]
            start = ((low + p - 1) // p) * p
            if start == p:
                start += p
            for j in range(start, high + 1, p):
                is_prime_seg[j - low] = False
        return [low + i for i, flag in enumerate(is_prime_seg) if flag]
    
    # Search larger primes in chunks
    search_limit = 5000000  # Try primes up to 5 million
    segment_size = 50000
    
    for seg_low in range(100000, search_limit, segment_size):
        if time.time() - start_time > deadline:
            break
        seg_high = min(seg_low + segment_size - 1, search_limit)
        
        seg_primes = sieve_segment(seg_low, seg_high, small_primes)
        
        # For each prime in this segment, check smoothness
        # Sort by some heuristic? Larger p might have longer runs
        # Actually, let's just check them all
        
        for p in seg_primes:
            if time.time() - start_time > deadline:
                break
            
            p2 = p * p
            k = 0
            for offset in range(1, 201):
                if time.time() - start_time > deadline:
                    break
                n = p2 + offset
                # Quick check: is n divisible by any small prime?
                # If not, and n > p, then n has a factor > p unless n is prime > p
                # Actually, we need full smoothness check
                if is_p_smooth(n, p, small_primes):
                    k = offset
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p
    
    # If time remains, try even larger primes
    if time.time() - start_time < deadline - 10:
        for seg_low in range(5000000, 10000000, 100000):
            if time.time() - start_time > deadline:
                break
            seg_high = min(seg_low + 100000 - 1, 10000000)
            
            seg_primes = sieve_segment(seg_low, seg_high, small_primes)
            
            for p in seg_primes:
                if time.time() - start_time > deadline:
                    break
                
                p2 = p * p
                k = 0
                for offset in range(1, 201):
                    if time.time() - start_time > deadline:
                        break
                    n = p2 + offset
                    if is_p_smooth(n, p, small_primes):
                        k = offset
                    else:
                        break
                if k > best_k:
                    best_k = k
                    best_p = p
    
    return {'p': best_p, 'k': best_k}
