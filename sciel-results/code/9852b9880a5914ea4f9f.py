import time
import math
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = 115  # safety margin before 120s kill

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

    # Sieve primes up to limit
    def sieve(limit):
        if limit < 2:
            return []
        is_p = bytearray([1]) * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                for j in range(i * i, limit + 1, i):
                    is_p[j] = 0
        return [i for i in range(2, limit + 1) if is_p[i]]

    # Check run length for a given prime p
    # Uses the modular trick: for each prime q <= p, find i in [1,k] where p^2+i ≡ 0 (mod q)
    def check_run_length(p, max_k=200, primes=None):
        if primes is None:
            primes = sieve(p) if p <= 100000 else sieve(min(p, 100000))
        
        p2 = p * p
        rem = [p2 + i for i in range(max_k + 1)]  # rem[i] = p^2 + i
        
        for q in primes:
            if q > p:
                break
            r = (-p2) % q
            if r == 0:
                r = q
            i = r
            while i <= max_k:
                while rem[i] % q == 0:
                    rem[i] //= q
                if rem[i] == 1:
                    # Early: this number is fully factored
                    pass
                i += q
        
        k = 0
        for i in range(1, max_k + 1):
            if rem[i] == 1:
                k += 1
            else:
                break
        return k

    # Fast check: just verify the first few numbers to quickly reject bad p
    def quick_check(p, test_count=5, primes=None):
        """Quickly check if the first test_count numbers after p^2 are p-smooth"""
        if primes is None:
            primes = sieve(min(p, 5000))
        
        p2 = p * p
        for i in range(1, test_count + 1):
            n = p2 + i
            # Remove small prime factors
            for q in primes:
                if q > p:
                    break
                if q * q > n:
                    break
                while n % q == 0:
                    n //= q
            if n > 1:
                # n has a factor > sqrt(original) or is prime > p
                if n > p:
                    return False
                # n is composite with all factors <= p? Check
                # If n <= p, it's fine (all its factors are <= n <= p)
                # If n > p, it must be prime > p (since we divided by all primes <= sqrt(n) <= sqrt(p^2+i) < p for small i)
                # Actually for n = p^2+i, sqrt(n) ≈ p, so primes up to p should suffice
                # But we only went up to min(p,5000), so for large p we might miss factors
                if n > p:
                    return False
        return True

    best_k = 0
    best_p = 2

    # Phase 1: Quick scan of small primes
    small_primes = sieve(10000)
    for p in small_primes:
        if time.time() - start_time > deadline - 30:
            break
        k = check_run_length(p, max_k=200, primes=small_primes)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: CRT-based targeted search
    # For k=1..10, try to construct p such that p^2+i has a small prime factor for each i
    # We use the fact that if p^2 ≡ -i (mod q_i) for some small prime q_i, then q_i | p^2+i
    # We need p to be prime and p^2+i to be fully p-smooth
    
    # Precompute useful small primes
    small_primes_for_crt = [q for q in small_primes if q <= 200]
    
    # Try CRT constructions for small target k
    for target_k in range(2, 15):
        if time.time() - start_time > deadline - 15:
            break
        
        # For each i in 1..target_k, pick a small prime q_i such that p^2 ≡ -i (mod q_i) has a solution
        # p^2 ≡ -i (mod q) has solution iff (-i/q) = 1 (Legendre symbol)
        # We'll try random combinations
        
        for attempt in range(20):
            if time.time() - start_time > deadline - 10:
                break
            
            # Pick random small primes for each i
            moduli = []
            residues = []
            valid = True
            for i in range(1, target_k + 1):
                # Find a small prime q where -i is a quadratic residue mod q
                found = False
                for q in small_primes_for_crt:
                    if q == 2:
                        # p^2 ≡ -i (mod 2): always solvable if i is odd, not if i even
                        # Actually p^2 mod 2 is 0 or 1
                        # -i mod 2: if i odd, -i ≡ 1, so p^2 ≡ 1 (mod 2) => p odd
                        # if i even, -i ≡ 0, so p^2 ≡ 0 (mod 2) => p even => p=2
                        if i % 2 == 1:
                            moduli.append(q)
                            residues.append(1)  # p ≡ 1 (mod 2)
                            found = True
                            break
                        else:
                            # p must be 2, which is too small for our purposes usually
                            continue
                    else:
                        # Check if -i is a quadratic residue mod q
                        # Use Euler's criterion: (-i)^(q-1)/2 ≡ 1 (mod q)
                        if pow(-i % q, (q - 1) // 2, q) == 1:
                            # Find a square root of -i mod q
                            # Simple: try all x in 1..q-1
                            for x in range(1, q):
                                if (x * x) % q == (-i) % q:
                                    moduli.append(q)
                                    residues.append(x)
                                    found = True
                                    break
                            if not found:
                                # Try -x
                                for x in range(1, q):
                                    if ((-x) * (-x)) % q == (-i) % q:
                                        moduli.append(q)
                                        residues.append(q - x)
                                        found = True
                                        break
                if not found:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Solve CRT: p ≡ residues[i] (mod moduli[i]) for i = 0..target_k-1
            # Use iterative CRT
            try:
                M = 1
                x = 0
                for r, m in zip(residues, moduli):
                    # Solve: x ≡ r (mod m), x ≡ current_x (mod current_M)
                    # x = current_x + t * current_M
                    # current_x + t * current_M ≡ r (mod m)
                    # t * current_M ≡ r - current_x (mod m)
                    g = gcd(current_M if 'current_M' in dir() else 1, m)
                    if g == 0:
                        current_M = m
                        current_x = r
                        continue
                    if 'current_M' not in dir():
                        current_M = m
                        current_x = r
                        continue
                    
                    g = gcd(current_M, m)
                    if (r - current_x) % g != 0:
                        break  # No solution
                    m1 = m // g
                    M1 = current_M // g
                    # t * M1 ≡ (r - current_x)/g (mod m1)
                    t = ((r - current_x) // g * pow(M1, -1, m1)) % m1
                    current_x = current_x + t * current_M
                    current_M = current_M * m1
                    x = current_x
                else:
                    # We have a solution x mod current_M
                    # Try a few values: x, x + current_M, x + 2*current_M, ...
                    # Check if any is prime and gives a good run
                    base = x % current_M
                    if base < 2:
                        base += current_M
                    
                    # Check a few candidates
                    for offset in range(5):
                        candidate = base + offset * current_M
                        if candidate > 10**9:
                            break
                        if candidate < 2:
                            continue
                        if not is_prime(candidate):
                            continue
                        # Quick check
                        k = check_run_length(candidate, max_k=200)
                        if k > best_k:
                            best_k = k
                            best_p = candidate
            except (ValueError, TypeError, ZeroDivisionError):
                continue

    # Phase 3: Wider sieve scan with early termination
    # Scan primes in ranges, using quick check first
    scan_limits = [50000, 100000, 200000, 500000]
    prev = 10000
    for limit in scan_limits:
        if time.time() - start_time > deadline - 10:
            break
        primes = sieve(limit)
        for p in primes:
            if p <= prev:
                continue
            if time.time() - start_time > deadline - 5:
                break
            # Quick check first
            if quick_check(p, test_count=3):
                k = check_run_length(p, max_k=200, primes=primes)
                if k > best_k:
                    best_k = k
                    best_p = p
        prev = limit

    # Phase 4: If we have time, try some larger primes with a different strategy
    # Use Pollard-Rho to factor the remaining parts
    if time.time() - start_time < deadline - 20:
        # Try primes around 10^6 to 10^7
        # Use a segmented sieve for primes in that range
        low = 10**6
        high = 2 * 10**6
        if high <= 10**9:
            # Simple sieve for this range
            size = high - low + 1
            is_prime_seg = bytearray([1]) * size
            small_primes_for_sieve = sieve(int(math.isqrt(high)) + 1)
            for q in small_primes_for_sieve:
                start = max(q * q, ((low + q - 1) // q) * q)
                for j in range(start - low, size, q):
                    is_prime_seg[j] = 0
            
            for i in range(size):
                if is_prime_seg[i]:
                    p = low + i
                    if time.time() - start_time > deadline - 5:
                        break
                    if quick_check(p, test_count=3):
                        # Full check with limited primes
                        primes_for_check = sieve(min(p, 50000))
                        k = check_run_length(p, max_k=200, primes=primes_for_check)
                        if k > best_k:
                            best_k = k
                            best_p = p

    # Final verification of the best result
    if best_k > 0:
        # Verify that p is prime
        if not is_prime(best_p):
            # Fallback to a known good small prime
            best_p = 2
            best_k = 1
        
        # Verify the run
        p2 = best_p * best_p
        primes_for_verify = sieve(min(best_p, 100000))
        valid = True
        for i in range(1, best_k + 1):
            n = p2 + i
            for q in primes_for_verify:
                if q > best_p:
                    break
                while n % q == 0:
                    n //= q
            if n > 1 and n > best_p:
                valid = False
                break
        
        if not valid:
            # Binary search for the actual valid k
            lo, hi = 1, best_k
            while lo < hi:
                mid = (lo + hi + 1) // 2
                valid_mid = True
                for i in range(1, mid + 1):
                    n = p2 + i
                    for q in primes_for_verify:
                        if q > best_p:
                            break
                        while n % q == 0:
                            n //= q
                    if n > 1 and n > best_p:
                        valid_mid = False
                        break
                if valid_mid:
                    lo = mid
                else:
                    hi = mid - 1
            best_k = lo

    return {'p': best_p, 'k': best_k}

if __name__ == "__main__":
    res = construct()
    print(res)
