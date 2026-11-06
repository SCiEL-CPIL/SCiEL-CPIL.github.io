"""
Strategy: Targeted Structural Search via CRT.
Instead of random sampling, we construct candidates n such that:
1. n is divisible by a large prime power P = p^a (a>=2).
2. n+1 is divisible by a large prime power Q = q^b (b>=2).

We iterate over pairs of significant prime powers (P, Q).
For each pair, we solve n ≡ 0 (mod P) and n ≡ -1 (mod Q).
This yields a specific residue class n ≡ r (mod LCM(P,Q)).
Since P and Q are prime powers of distinct primes (mostly), LCM(P,Q) = P*Q.
We then check all n in the range [10000, 999999] satisfying this congruence.
This ensures powerful_part(n) >= P and powerful_part(n+1) >= Q.
We filter for large P and Q to maximize the product P*Q.
"""

import time
import math
import sys
import numpy as np

# EVOLVE-BLOCK-START

def _build_spf(limit):
    """Build smallest-prime-factor sieve up to limit."""
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _powerful_part_fast(n, spf):
    """Use SPF table if available; fall back to trial division."""
    if spf is not None and n < len(spf):
        pp = 1
        x = n
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                pp *= p ** e
        return pp
    else:
        # Fallback trial division
        pp = 1
        d = 2
        while d * d <= n:
            if n % d == 0:
                e = 0
                while n % d == 0:
                    n //= d
                    e += 1
                if e >= 2:
                    pp *= d ** e
            d += 1 if d == 2 else 2
        if n > 1:
            pass
        return pp

def _score_candidate(n, spf):
    """Return powerful_part(n) * powerful_part(n+1)."""
    if n < 10000 or n + 1 > 1000000:
        return 0
    ppn = _powerful_part_fast(n, spf)
    ppn1 = _powerful_part_fast(n + 1, spf)
    return ppn * ppn1

def _get_prime_powers(limit):
    """
    Generate list of prime powers p^e <= limit where e >= 2.
    Returns list of (value, prime, exponent).
    """
    powers = []
    # We only need primes up to sqrt(limit) for e>=2, but p^2 can be up to limit.
    # Actually p can be up to sqrt(limit).
    # Let's just iterate primes up to 1000 (sqrt(1e6)) and generate powers.
    primes = []
    is_prime = [True] * (int(math.isqrt(limit)) + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, int(math.isqrt(limit)) + 1, i):
                is_prime[j] = False
            
    for p in primes:
        pk = p * p
        e = 2
        while pk <= limit:
            powers.append((pk, p, e))
            pk *= p
            e += 1
            
    # Also include prime powers where p > sqrt(limit) but p^2 <= limit?
    # No, if p > sqrt(limit), then p^2 > limit. So all bases are <= sqrt(limit).
    
    # Sort by value descending to prioritize large powers
    powers.sort(key=lambda x: x[0], reverse=True)
    return powers

def _search_crt(deadline, spf):
    """
    Structural search using CRT.
    """
    best_n = 10000
    best_score = _score_candidate(10000, spf)
    
    # Generate prime powers up to 1000000
    prime_powers = _get_prime_powers(1000000)
    
    # Filter to keep only "significant" powers to reduce pair combinations.
    # We want P and Q to be large.
    # Let's take the top 100 prime powers.
    top_powers = prime_powers[:100]
    
    # To avoid redundant checks (P,Q) and (Q,P) are different constraints but we check n and n+1,
    # actually n % P == 0 and n+1 % Q == 0 is distinct from n % Q == 0 and n+1 % P == 0.
    # So we iterate all pairs.
    
    # Optimization: Precompute modular inverses? 
    # n = k * P.
    # k * P + 1 ≡ 0 (mod Q)  => k * P ≡ -1 (mod Q).
    # If gcd(P, Q) != 1, no solution.
    # If gcd(P, Q) == 1, k ≡ -1 * P^{-1} (mod Q).
    
    # We will iterate P from top_powers, and Q from top_powers.
    # To save time, we can break early if P * Q is small? No, we want large.
    # But we must check if solutions exist in range.
    
    # Let's limit the number of pairs to check to fit in time.
    # 100 * 100 = 10,000 pairs. For each, finding k is O(1) with pow(P, -1, Q).
    # Then checking the specific n in range is O(1) or O(few).
    # This is very fast.
    
    for i, (P, p, e) in enumerate(top_powers):
        if time.time() > deadline:
            break
        for j, (Q, q, f) in enumerate(top_powers):
            if time.time() > deadline:
                break
            
            # Skip if same prime base? 
            # If p == q, then gcd(P, Q) > 1. 
            # n is multiple of p^max(e,f). n+1 is multiple of p^min(e,f)? No, n+1 not divisible by p.
            # So if p == q, there is NO solution for n % p^e == 0 and n+1 % p^f == 0 (since p^f | n+1 implies p | n+1, so p ∤ n).
            # So we can skip pairs with same prime base.
            if p == q:
                continue
            
            # Solve k * P ≡ -1 (mod Q)
            # Since gcd(P, Q) = 1 (distinct primes), inverse exists.
            try:
                inv_P = pow(P, -1, Q)
            except ValueError:
                continue # Should not happen if p != q
            
            k_mod = (-1 * inv_P) % Q
            
            # We need n = k * P such that 10000 <= n <= 999999 and k ≡ k_mod (mod Q).
            # k = k_mod + m * Q for m = 0, 1, ...
            # n = P * (k_mod + m * Q)
            
            # Find range of m
            # 10000 <= P * (k_mod + m * Q) <= 999999
            # 10000 / P <= k_mod + m * Q <= 999999 / P
            
            min_k = math.ceil(10000 / P)
            max_k = math.floor(999999 / P)
            
            if min_k > max_k:
                continue
            
            # Find smallest m >= 0 such that k_mod + m * Q >= min_k
            # m * Q >= min_k - k_mod
            if k_mod >= min_k:
                m_start = 0
            else:
                m_start = math.ceil((min_k - k_mod) / Q)
                
            # Find largest m such that k_mod + m * Q <= max_k
            # m * Q <= max_k - k_mod
            if k_mod <= max_k:
                m_end = math.floor((max_k - k_mod) / Q)
            else:
                continue # No solution
            
            if m_start > m_end:
                continue
            
            # Check a few candidates in this arithmetic progression.
            # Usually there is only one or a few.
            # We check the first one and the last one.
            candidates_k = set()
            candidates_k.add(k_mod + m_start * Q)
            candidates_k.add(k_mod + m_end * Q)
            # Maybe middle?
            if m_end - m_start > 10:
                 candidates_k.add(k_mod + ((m_start + m_end)//2) * Q)
            
            for k in candidates_k:
                n = P * k
                if 10000 <= n <= 999999:
                    s = _score_candidate(n, spf)
                    if s > best_score:
                        best_score = s
                        best_n = n
                        
    return best_n, best_score

def _search_random(n_trials, spf, deadline):
    """Random sampling in [10000, 999999]."""
    import random
    best_n = 10000
    best_score = _score_candidate(10000, spf)
    for _ in range(n_trials):
        if time.time() > deadline:
            break
        n = random.randint(10000, 999999)
        s = _score_candidate(n, spf)
        if s > best_score:
            best_score = s
            best_n = n
    return best_n, best_score

def construct():
    start_time = time.time()
    total_deadline = start_time + 85  # 5s margin
    
    # Build SPF for fast factorization up to 1,000,001
    spf_limit = 1000001
    spf = _build_spf(spf_limit)
    
    best_n = 10000
    best_score = _score_candidate(10000, spf)
    
    # Phase 1: CRT Structural Search (High priority as it targets high score structures)
    if time.time() < total_deadline - 5:
        cn, cs = _search_crt(total_deadline, spf)
        if cs > best_score:
            best_score = cs
            best_n = cn
            
    # Phase 2: Random sampling to catch cases where powerful parts come from multiple small primes
    # or where the CRT constraint was too strict (e.g. n has p^2 * q^2, n+1 has r^2 * s^2).
    # The CRT search guarantees n is div by P and n+1 by Q, but doesn't prevent n from having MORE powerful part.
    # However, random sampling might find n where n is not div by a single large P but product of two medium ones.
    
    remaining = total_deadline - time.time()
    if remaining > 2:
        n_trials = int(min(50000, remaining * 5000))
        n_trials = max(n_trials, 1000)
        rn, rs = _search_random(n_trials, spf, total_deadline)
        if rs > best_score:
            best_score = rs
            best_n = rn
            
    return best_n

# EVOLVE-BLOCK-END
