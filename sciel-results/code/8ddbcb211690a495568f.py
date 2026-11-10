import numpy as np
import time
import math
from fractions import Fraction

def _sigma(n):
    """Compute sigma(n) - sum of all positive divisors of n."""
    if n == 1:
        return 1
    result = 1
    m = n
    # Factor out powers of 2
    if m % 2 == 0:
        p_sum = 1
        p_pow = 1
        while m % 2 == 0:
            m //= 2
            p_pow *= 2
            p_sum += p_pow
        result *= p_sum
    # Try odd primes
    p = 3
    while p * p <= m:
        if m % p == 0:
            p_sum = 1
            p_pow = 1
            while m % p == 0:
                m //= p
                p_pow *= p
                p_sum += p_pow
            result *= p_sum
        p += 2
    if m > 1:
        result *= (1 + m)
    return result

def _factorize_sigma(n):
    """Factorize n into prime powers, return list of (p, e)."""
    factors = []
    m = n
    if m % 2 == 0:
        e = 0
        while m % 2 == 0:
            m //= 2
            e += 1
        factors.append((2, e))
    p = 3
    while p * p <= m:
        if m % p == 0:
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            factors.append((p, e))
        p += 2
    if m > 1:
        factors.append((m, 1))
    return factors

def _sigma_from_factors(factors):
    """Compute sigma from prime factorization [(p, e), ...]."""
    result = 1
    for p, e in factors:
        s = 1
        for _ in range(e):
            s = s * p + 1
        result *= s
    return result

def construct():
    N = 10_000_000
    deadline = time.time() + 80  # Safety margin
    
    # Strategy: For each small k, look for pairs where 2a - 3b = k
    # This gives a/b = 3/2 + k/(2b), so error = |k|/(2b)
    # We want large b and small |k|
    #
    # For a given k, b = (2a - k)/3, so a = (3b + k)/2
    # We need 3b + k to be even, so k must have same parity as 3b
    # i.e., k must have same parity as b (since 3b has same parity as b)
    #
    # Also need a, b <= N, so (3b+k)/2 <= N => b <= 2(N-k)/3 (for k>0)
    # and b >= max(1, -k/2) for a >= 1
    #
    # We'll iterate over small |k| values and large b values,
    # computing sigma(a) and sigma(b) on the fly.
    
    best_a, best_b = 11, 6
    best_num = abs(2*11 - 3*6)  # = 4
    best_den = 2*6  # = 12
    
    # Cache sigma values to avoid recomputation
    sigma_cache = {}
    
    def get_sigma(n):
        if n in sigma_cache:
            return sigma_cache[n]
        s = _sigma(n)
        sigma_cache[n] = s
        return s
    
    # Try small |k| values first (they give smallest error for large b)
    # For each k, try large b values
    max_k = 200  # Max absolute value of 2a-3b to try
    
    for k in range(0, max_k + 1):
        if time.time() > deadline - 10:
            break
        
        # k = 0: a/b = 3/2 exactly, but a = 3b/2 must be integer
        # So b must be even: b = 2t, a = 3t
        # Check if sigma(3t) = sigma(2t)
        if k == 0:
            # b must be even, a = 3b/2
            # b ranges from 2 to 2N/3
            # Try large b values
            max_b = (2 * N) // 3
            # Try b values: start from large, go down
            # b must be even
            for b in range(max_b - (max_b % 2), 0, -2):
                if time.time() > deadline - 5:
                    break
                a = 3 * b // 2
                if a > N or a <= 0:
                    continue
                sa = get_sigma(a)
                sb = get_sigma(b)
                if sa == sb:
                    # Perfect! Error is 0, but that's invalid
                    # Actually the task says raw value of exactly 0 is invalid
                    # So skip this
                    pass
                # We found a match but error is 0, which is invalid
                # Move on to k >= 1
            continue
        
        # For k > 0: a = (3b + k)/2
        # Need 3b + k even => b and k same parity
        # a <= N => 3b + k <= 2N => b <= (2N - k)/3
        # a >= 1 => 3b + k >= 2 => b >= (2-k)/3
        # b >= 1
        
        max_b = (2 * N - k) // 3
        min_b = max(1, (2 - k + 2) // 3)  # ceil((2-k)/3) for k<2, else 1
        
        # b must have same parity as k
        start_b = max_b
        if start_b % 2 != k % 2:
            start_b -= 1
        
        # Try b values from large to small
        # For efficiency, try a subset first
        step = 1
        found = False
        for b in range(start_b, min_b - 1, -step):
            if time.time() > deadline - 5:
                break
            a = (3 * b + k) // 2
            if a > N or a < 1:
                continue
            sa = get_sigma(a)
            sb = get_sigma(b)
            if sa == sb:
                # Found a match!
                # Error = k/(2b)
                num = k
                den = 2 * b
                if num * best_den < best_num * den:
                    best_num = num
                    best_den = den
                    best_a, best_b = a, b
                    found = True
                    # Keep searching for better (larger b) in this k
                    # But we're going from large b to small, so first found is best for this k
                    # Actually we want larger b, so break after first found for this k
                    break
        
        # Also try negative k: 2a - 3b = -k => a = (3b - k)/2
        # Need 3b - k even => b and k same parity
        # a <= N => 3b - k <= 2N => b <= (2N + k)/3
        # a >= 1 => 3b - k >= 2 => b >= (k+2)/3
        min_b_neg = max(1, (k + 2 + 2) // 3)  # ceil((k+2)/3)
        max_b_neg = (2 * N + k) // 3
        
        start_b_neg = max_b_neg
        if start_b_neg % 2 != k % 2:
            start_b_neg -= 1
        
        for b in range(start_b_neg, min_b_neg - 1, -step):
            if time.time() > deadline - 5:
                break
            a = (3 * b - k) // 2
            if a > N or a < 1:
                continue
            sa = get_sigma(a)
            sb = get_sigma(b)
            if sa == sb:
                # Found a match!
                # Error = k/(2b)
                num = k
                den = 2 * b
                if num * best_den < best_num * den:
                    best_num = num
                    best_den = den
                    best_a, best_b = a, b
                break
    
    return [int(best_a), int(best_b)]
