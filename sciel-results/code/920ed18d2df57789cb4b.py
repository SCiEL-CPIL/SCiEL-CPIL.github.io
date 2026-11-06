import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Deterministic Miller-Rabin for 64-bit
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def is_prime(n):
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in MR_BASES:
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
    
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randrange(1, n)
            c = random.randrange(1, n)
            m = random.randrange(1, n)
            g, q, r = 1, 1, 1
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
            if g != n:
                return g
    
    def all_factors_leq(n, limit):
        """Check if all prime factors of n are <= limit."""
        if n <= 1:
            return True
        # Trial division by small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for p in small_primes:
            if n % p == 0:
                if p > limit:
                    return False
                while n % p == 0:
                    n //= p
                if n == 1:
                    return True
                if is_prime(n):
                    return n <= limit
                break
        
        if is_prime(n):
            return n <= limit
        
        # Pollard Brent
        d = pollard_brent(n)
        return all_factors_leq(d, limit) and all_factors_leq(n // d, limit)
    
    def check_run_full(p):
        """Full factorization check for p^2+1 .. p^2+k."""
        k = 0
        base = p * p
        while k < 200:
            n = base + k + 1
            if not all_factors_leq(n, p):
                break
            k += 1
        return k
    
    # Precompute forbidden residues for small primes q
    # For each i in [1, 200] and each prime q, find r such that r^2 ≡ -i (mod q)
    # If such r exists, then p ≡ r (mod q) means p^2 + i ≡ 0 (mod q), so q divides p^2+i
    # If q > p, then p^2+i has a prime factor > p, so p is invalid for this i
    
    # Choose a limit for small q to precompute
    Q_LIMIT = 100000  # Precompute for primes q up to 100000
    
    # Sieve primes up to Q_LIMIT
    def sieve_primes(limit):
        if limit < 2:
            return []
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i in range(2, limit + 1) if sieve[i]]
    
    small_primes_list = sieve_primes(Q_LIMIT)
    
    # For each i in [1, 200] and each prime q in small_primes_list,
    # compute the set of residues r mod q where r^2 ≡ -i (mod q)
    # Store as: forbidden[i][q] = set of residues
    
    # To make checking fast, for each q, precompute for all i at once
    # Actually, for a given p, we check: for each i in [1, k], for each q in small_primes_list
    # where q > p, check if p mod q is in forbidden[i][q]
    
    # But q > p is important. If q <= p, then even if q divides p^2+i, it's fine since q <= p.
    # So we only care about q > p.
    
    # Precompute: for each q in small_primes_list, and for each i in [1, 200],
    # the residues where r^2 ≡ -i (mod q)
    
    # To save memory and time, let's precompute a dictionary:
    # q_to_forbidden[q] = list of (i, r1, r2) where r1^2 ≡ -i (mod q)
    
    # Actually, for checking, for a given p and i, we need to check if there exists
    # a prime q > p (q in small_primes_list) such that p mod q is a root of x^2 ≡ -i (mod q).
    
    # Let's precompute for each q: for each i, the roots.
    # But 200 * 100000 = 20 million entries, might be too much.
    
    # Alternative: For each candidate p, for each i in [1, current_k], check small q.
    # But that's 200 * (number of small primes > p) checks per p.
    
    # Let's try a different approach: For a block of p values, sieve.
    
    # Actually, let's just do: for candidate p, check small q constraints quickly,
    # then full factorize if it passes.
    
    # Precompute for each q in small_primes_list, a dict from i to list of roots
    q_forbidden = {}
    for q in small_primes_list:
        q_forbidden[q] = {}
        for i in range(1, 201):
            target = (-i) % q
            # Check if target is a quadratic residue mod q
            # Since q is prime, use Euler's criterion or try to find sqrt
            # For small q, we can just try all r in [0, q-1]
            # But q up to 100000, that's too slow for 200 * 100000 * 100000
            # Better: use pow(target, (q-1)//2, q) to check if residue
            # If residue, find sqrt using Tonelli-Shanks or brute force for small q
            
            # For efficiency, let's only precompute for q up to a smaller limit
            # and for larger q, skip in the quick check
    
    # This is getting complex. Let me simplify:
    # Use a smaller Q_LIMIT for precomputation, say 10000
    # And for the quick check, only use primes q in (p, Q_LIMIT]
    
    Q_LIMIT = 10000
    small_primes_list = sieve_primes(Q_LIMIT)
    
    # Precompute: for each q in small_primes_list, for each i in [1, 200],
    # the list of r in [0, q-1] such that r^2 ≡ -i (mod q)
    # Store as: q_roots[q][i] = list of roots
    
    q_roots = {}
    for q in small_primes_list:
        q_roots[q] = {}
        for i in range(1, 201):
            target = (-i) % q
            roots = []
            # Try to find sqrt of target mod q
            # For q up to 10000, brute force is feasible: q iterations per (q, i)
            # Total: sum_{q in primes up to 10000} 200 * q ≈ 200 * (sum of primes up to 10000)
            # Sum of primes up to 10000 ≈ 5.7 million, so 200 * 5.7M = 1.14 billion, too slow
            
            # Better: use pow to check if residue, then use Tonelli-Shanks
            if q == 2:
                if target == 0:
                    roots = [0]
                elif target == 1:
                    roots = [1]
                else:
                    roots = []
            else:
                # Check if target is a quadratic residue
                if target == 0:
                    roots = [0]
                else:
                    # Euler's criterion
                    if pow(target, (q - 1) // 2, q) == 1:
                        # It's a residue, find sqrt
                        # Tonelli-Shanks
                        def tonelli_shanks(n, p):
                            if p % 4 == 3:
                                return pow(n, (p + 1) // 4, p)
                            # General case
                            q_val = p - 1
                            s = 0
                            while q_val % 2 == 0:
                                q_val //= 2
                                s += 1
                            # Find z such that z^2 ≡ -1 (mod p)
                            z = 2
                            while pow(z, (p - 1) // 2, p) != p - 1:
                                z += 1
                            m = s
                            c = pow(z, q_val, p)
                            t = pow(n, q_val, p)
                            r = pow(n, (q_val + 1) // 2, p)
                            while t != 1:
                                i = 0
                                t2 = t
                                while t2 != 1:
                                    t2 = (t2 * t2) % p
                                    i += 1
                                b = pow(c, 1 << (m - i - 1), p)
                                m = i
                                c = (b * b) % p
                                t = (t * c) % p
                                r = (r * b) % p
                            return r
                        root1 = tonelli_shanks(target, q)
                        root2 = (q - root1) % q
                        roots = [root1, root2] if root1 != root2 else [root1]
                    else:
                        roots = []
            q_roots[q][i] = roots
    
    # Now, for a candidate p, quick check:
    # For each i in [1, 200], for each prime q in small_primes_list where q > p,
    # check if p mod q is in q_roots[q][i]
    # If yes, then p^2 + i ≡ 0 (mod q) and q > p, so p^2 + i has a prime factor > p
    # So p is invalid for this i
    
    def quick_check(p, max_i=200):
        """Return the maximum k such that for all i in [1, k], p^2+i is not divisible by any prime q in (p, Q_LIMIT]."""
        k = 0
        for i in range(1, max_i + 1):
            bad = False
            for q in small_primes_list:
                if q <= p:
                    continue
                if q > Q_LIMIT:
                    break
                p_mod_q = p % q
                if p_mod_q in q_roots[q][i]:
                    bad = True
                    break
            if bad:
                break
            k += 1
        return k
    
    # Generate candidate primes up to some limit
    # Use sieve up to 10^7
    SIEVE_LIMIT = 10**7
    primes = sieve_primes(SIEVE_LIMIT)
    
    best_k = 0
    best_p = 2
    
    # For each prime p, do quick check, and if it looks promising, do full check
    for p in primes:
        if time.time() > deadline:
            break
        
        # Quick check
        k_quick = quick_check(p)
        
        # If k_quick is 0, skip
        if k_quick == 0:
            continue
        
        # If k_quick is high, do full check
        # Only do full check if k_quick > best_k or if it might improve
        if k_quick > best_k:
            k_full = check_run_full(p)
            if k_full > best_k:
                best_k = k_full
                best_p = p
        
        # Also, even if k_quick <= best_k, the full check might give a higher k
        # because quick check only uses small q. But if k_quick is low, full k is likely low too.
        # To be safe, do full check if k_quick >= best_k - 5
        elif k_quick >= best_k - 5:
            k_full = check_run_full(p)
            if k_full > best_k:
                best_k = k_full
                best_p = p
    
    # If time remains, try some larger primes using Miller-Rabin
    if time.time() < deadline - 10:
        # Try random large primes
        for _ in range(100):
            if time.time() > deadline:
                break
            p = random.randrange(10**7, 10**9)
            # Make it odd
            p |= 1
            if not is_prime(p):
                continue
            k_quick = quick_check(p)
            if k_quick > 0:
                k_full = check_run_full(p)
                if k_full > best_k:
                    best_k = k_full
                    best_p = p
    
    return {'p': best_p, 'k': best_k}
