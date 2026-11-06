"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt
from collections import Counter

def _factorize(n):
    """Trial division factorization for n up to ~10^18 (slow but works for structured inputs)."""
    if n == 1:
        return []
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            cnt = 0
            while n % d == 0:
                cnt += 1
                n //= d
            factors.append((d, cnt))
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append((n, 1))
    return factors

def _is_powerful(n):
    """Check if n is powerful using factorization."""
    if n == 1:
        return True
    factors = _factorize(n)
    for p, e in factors:
        if e < 2:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    best_N = None
    best_count = 0
    best_pairs = []
    
    # Strategy: N = R * M where R is powerful and M has many sum-of-two-squares reps.
    # For each rep u^2 + v^2 = M with u <= v, we get a = R*u^2, b = R*v^2.
    # Both a and b are powerful if R is powerful and u, v are positive integers.
    # (R*p^k has all prime exponents >= 2 when R is powerful.)
    
    # Key insight: maximize M's representation count while keeping R*M <= 10^18.
    # M should be of form 2^(2f) * prod(p_i^(2e_i)) where p_i = 4k+1 primes.
    # r_2(M) = 4 * prod(2*e_i + 1) if all 4k+3 prime exponents are even.
    
    # Try different R values (powerful) and for each, find optimal M.
    
    # Small powerful R values to try
    R_candidates = [1]
    # p^3 for small primes
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        R_candidates.append(p**3)
    # Products of small p^3
    R_candidates.append(8 * 27)        # 216
    R_candidates.append(8 * 125)       # 1000
    R_candidates.append(27 * 125)      # 3375
    R_candidates.append(8 * 27 * 125)  # 27000
    R_candidates.append(125 * 343)     # 42875
    R_candidates.append(8 * 27 * 125 * 343)  # 9261000
    R_candidates.append(8 * 1331)      # 10648
    R_candidates.append(27 * 1331)     # 35937
    R_candidates = sorted(set(R_candidates))
    
    # 4k+1 primes for M construction
    primes_4k1 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257, 269, 277, 281, 293, 313, 317]
    
    for R in R_candidates:
        if time.time() > deadline - 30:
            break
        
        M_max = 10**18 // R
        if M_max < 2:
            continue
        
        # Find M <= M_max of form 2^(2f) * prod(p_i^(2e_i)) with max rep count.
        # We'll try M = (prod of subset of primes_4k1)^2 * 4^f
        # Rep count = 4 * prod(2*e_i + 1) where e_i is exponent of p_i in the product.
        # For M = (p1*p2*...*pk)^2, rep count = 4 * 3^k.
        
        best_M_for_R = 1
        best_reps_for_R = 1
        best_reps_list_for_R = []
        
        # Try M = (prod of first k primes_4k1)^2 for k = 0, 1, 2, ...
        product = 1
        for k, p in enumerate(primes_4k1):
            new_product = product * p
            M_cand = new_product * new_product
            if M_cand > M_max:
                break
            
            # Count representations of M_cand as x^2 + y^2
            reps = []
            x = 0
            while x * x <= M_cand // 2:
                y2 = M_cand - x * x
                y = isqrt(y2)
                if y * y == y2 and x <= y:
                    reps.append((x, y))
                x += 1
            
            cnt = len(reps)
            if cnt > best_reps_for_R:
                best_reps_for_R = cnt
                best_M_for_R = M_cand
                best_reps_list_for_R = reps
            
            product = new_product
        
        # Also try M = 4 * (prod of first k primes_4k1)^2
        product = 1
        for k, p in enumerate(primes_4k1):
            new_product = product * p
            M_cand = 4 * new_product * new_product
            if M_cand > M_max:
                break
            
            reps = []
            x = 0
            while x * x <= M_cand // 2:
                y2 = M_cand - x * x
                y = isqrt(y2)
                if y * y == y2 and x <= y:
                    reps.append((x, y))
                x += 1
            
            cnt = len(reps)
            if cnt > best_reps_for_R:
                best_reps_for_R = cnt
                best_M_for_R = M_cand
                best_reps_list_for_R = reps
            
            product = new_product
        
        # Evaluate N = R * best_M_for_R
        N = R * best_M_for_R
        if N < 2 or N > 10**18:
            continue
        
        if best_reps_for_R > best_count:
            best_count = best_reps_for_R
            best_N = N
            # Generate pairs: a = R*u^2, b = R*v^2 for each (u,v) in best_reps_list_for_R
            pairs = []
            seen = set()
            for u, v in best_reps_list_for_R:
                a = R * u * u
                b = R * v * v
                if a > b:
                    a, b = b, a
                if a + b == N and (a, b) not in seen:
                    seen.add((a, b))
                    pairs.append([a, b])
            best_pairs = pairs
    
    # Verify best result
    if best_N is not None and best_count > 0:
        # Double-check a few pairs are actually powerful
        verified_pairs = []
        for a, b in best_pairs:
            if a + b == best_N and a <= b and a >= 1 and b >= 1:
                if _is_powerful(a) and _is_powerful(b):
                    verified_pairs.append([a, b])
        
        if len(verified_pairs) > 0:
            return {'N': best_N, 'pairs': verified_pairs}
    
    # Fallback: simple valid answer
    return {'N': 2, 'pairs': [[1, 1]]}
# EVOLVE-BLOCK-END
