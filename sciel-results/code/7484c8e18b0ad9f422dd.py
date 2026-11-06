"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

DEADLINE = 85.0
_start = time.time()

def _elapsed():
    return time.time() - _start

def _sieve_spf(max_n):
    """Build smallest prime factor array up to max_n."""
    spf = np.arange(max_n + 1, dtype=np.int64)
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(math.isqrt(max_n)) + 1):
        if spf[i] == i:
            spf[i*i::i] = np.where(spf[i*i::i] == np.arange(i*i, max_n+1, i), i, spf[i*i::i])
    return spf

def _precompute_powerful_parts(max_n, spf):
    """Precompute powerful part for all numbers 1..max_n using SPF with vectorized recurrence."""
    pp = np.ones(max_n + 1, dtype=np.int64)
    # For each n, let p = spf[n], m = n // p.
    # If spf[m] == p, then p's exponent in n is (exp in m) + 1.
    #   - If exp in m >= 2: pp[n] = pp[m] * p (the p^e factor grows)
    #   - If exp in m == 1: pp[n] = pp[m] (p^1 is not powerful, but m had no p^2+ either... wait)
    #   - If exp in m == 0: impossible since spf[m] == p means p divides m.
    # If spf[m] != p (or m == 1), then p appears with exponent exactly 1 in n, so pp[n] = pp[m].
    #
    # More carefully:
    # Let n = p^e * r where gcd(p,r)=1.
    # pp[n] = p^e if e>=2, else 1, times pp[r].
    # m = n/p = p^(e-1) * r.
    # pp[m] = p^(e-1) if e-1>=2 (i.e., e>=3), else 1 if e==2, times pp[r].
    #
    # Case e >= 3: pp[m] = p^(e-1) * pp[r], pp[n] = p^e * pp[r] = pp[m] * p
    # Case e == 2: pp[m] = pp[r] (since e-1=1 < 2), pp[n] = p^2 * pp[r] = pp[m] * p^2
    # Case e == 1: pp[m] = pp[r] (m = r, p not in m), pp[n] = pp[r] = pp[m]
    #
    # So: if spf[m] == p (meaning e >= 2 in n, i.e., e-1 >= 1 in m):
    #   We need to know if e >= 3 or e == 2.
    #   e >= 3 iff spf[m/p] == p (i.e., p still divides m/p)
    #   Simpler: if spf[m] == p, check if m//p still has spf == p.
    #
    # Vectorized approach:
    # For each n from 2 to max_n:
    #   p = spf[n], m = n // p
    #   if m >= 2 and spf[m] == p:
    #       # p divides m, so e >= 2 in n
    #       m2 = m // p
    #       if m2 >= 2 and spf[m2] == p:
    #           # e >= 3
    #           pp[n] = pp[m] * p
    #       else:
    #           # e == 2
    #           pp[n] = pp[m] * p * p
    #   else:
    #       # e == 1 (or n is prime)
    #       pp[n] = pp[m]
    
    # Vectorize with numpy
    n_arr = np.arange(2, max_n + 1)
    p_arr = spf[n_arr]
    m_arr = n_arr // p_arr
    
    # Determine which m values have spf[m] == p
    # m_arr can be 1 (when n is prime)
    valid_m = m_arr >= 2
    # For valid_m, check spf[m] == p
    # We need to be careful: spf is indexed up to max_n, m_arr values are <= max_n//2
    
    # Initialize result
    result = np.ones(max_n + 1, dtype=np.int64)
    
    # For all n where m < 2 or spf[m] != p: pp[n] = pp[m]
    # For n where m >= 2 and spf[m] == p: need further check
    
    # Step 1: compute m_arr and p_arr
    # Step 2: categorize
    # Let's do it in chunks to avoid huge temporary arrays if needed, but max_n=1e6 should be fine
    
    # m_arr values: for n=2..max_n, m = n//spf[n]
    # For n=2: p=2, m=1
    # For n=3: p=3, m=1
    # For n=4: p=2, m=2
    # For n=6: p=2, m=3
    # etc.
    
    # Build the recurrence
    # pp[n] depends on pp[m] where m < n, so we can't fully vectorize without iteration.
    # But we can process in order. Let's use a hybrid: vectorize the simple cases.
    
    # Actually, let's just use a faster Python loop with local variable caching
    pp_local = pp
    spf_local = spf
    deadline = DEADLINE
    
    for n in range(2, max_n + 1):
        p = spf_local[n]
        m = n // p
        if m < 2:
            pp_local[n] = 1
        elif spf_local[m] == p:
            m2 = m // p
            if m2 < 2 or spf_local[m2] != p:
                # e == 2
                pp_local[n] = pp_local[m] * p * p
            else:
                # e >= 3
                pp_local[n] = pp_local[m] * p
        else:
            pp_local[n] = pp_local[m]
        if _elapsed() > deadline:
            break
    
    return pp

def _generate_powerful_numbers(max_val):
    """Generate all powerful numbers (products of distinct prime powers p^e, e>=2) up to max_val."""
    limit = int(math.isqrt(max_val)) + 2
    if limit < 2:
        limit = 2
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, limit + 1) if sieve[i]]
    
    prime_powers = {}
    for p in primes:
        powers = []
        val = p * p
        e = 2
        while val <= max_val and e <= 12:
            powers.append(val)
            e += 1
            val *= p
        prime_powers[p] = powers
    
    active_primes = [p for p in primes if p * p <= max_val]
    
    current = [1]
    for p in active_primes:
        if _elapsed() > DEADLINE:
            break
        powers = prime_powers.get(p, [])
        if not powers:
            continue
        new_current = list(current)
        for pw in powers:
            for v in current:
                prod = v * pw
                if prod <= max_val:
                    new_current.append(prod)
        current = new_current
    
    powerful_list = sorted(set(current))
    return powerful_list

def _solve_crt_zero_minus1(A, B):
    """Solve x ≡ 0 (mod A), x ≡ -1 (mod B). Returns (x0, lcm) or None."""
    g = math.gcd(A, B)
    if (B - 1) % g != 0:
        return None
    a1 = A // g
    b1 = B // g
    c1 = (B - 1) // g
    inv_a1 = pow(a1, -1, b1)
    k0 = (c1 * inv_a1) % b1
    x0 = A * k0
    lcm = A * b1
    return x0, lcm

def construct():
    global _start, DEADLINE
    _start = time.time()
    DEADLINE = 85.0
    
    best_score = 0
    best_n = 10000
    
    max_n = 1000000
    min_n = 10000
    
    # Build SPF
    spf = _sieve_spf(max_n + 2)
    if _elapsed() > DEADLINE:
        return 10000
    
    # Precompute powerful parts for all numbers up to max_n+1
    pp = _precompute_powerful_parts(max_n + 1, spf)
    if _elapsed() > DEADLINE:
        return 10000
    
    # Direct scan of all consecutive pairs - this finds the GLOBAL OPTIMUM
    # pp[n] * pp[n+1] for n in [min_n, max_n]
    scores = pp[min_n:max_n+1] * pp[min_n+1:max_n+2]
    idx = int(np.argmax(scores))
    best_score = int(scores[idx])
    best_n = min_n + idx
    
    if _elapsed() > DEADLINE:
        return best_n
    
    # Generate powerful numbers for CRT-based refinement
    powerful_nums = _generate_powerful_numbers(max_n)
    if _elapsed() > DEADLINE:
        return best_n
    
    powerful_nums = [x for x in powerful_nums if x > 1]
    powerful_nums.sort(reverse=True)
    
    # CRT-based refinement: try to find n where powerful_part(n) >= A and powerful_part(n+1) >= B
    # with A*B > best_score, using top powerful numbers
    K = min(len(powerful_nums), 500)
    candidates = powerful_nums[:K]
    
    for i, A in enumerate(candidates):
        if _elapsed() > DEADLINE - 2.0:
            break
        for j, B in enumerate(candidates):
            if _elapsed() > DEADLINE - 2.0:
                break
            if i == j:
                continue
            if A * B <= best_score:
                continue
            if math.gcd(A, B) != 1:
                continue
            
            sol = _solve_crt_zero_minus1(A, B)
            if sol is None:
                continue
            x0, lcm = sol
            
            if x0 < min_n:
                t = (min_n - x0 + lcm - 1) // lcm
                n_val = x0 + t * lcm
            else:
                n_val = x0
            
            if n_val < min_n or n_val > max_n:
                continue
            
            actual_score = int(pp[n_val] * pp[n_val + 1])
            if actual_score > best_score:
                best_score = actual_score
                best_n = n_val
    
    # Local search around best_n: check neighbors for potentially better scores
    # (the full scan should have caught these, but verify)
    if _elapsed() < DEADLINE - 1.0:
        search_radius = 100
        lo = max(min_n, best_n - search_radius)
        hi = min(max_n, best_n + search_radius)
        local_scores = pp[lo:hi+1] * pp[lo+1:hi+2]
        local_idx = int(np.argmax(local_scores))
        local_score = int(local_scores[local_idx])
        if local_score > best_score:
            best_score = local_score
            best_n = lo + local_idx
    
    return best_n

# EVOLVE-BLOCK-END
