import time
import math

def _factorize(num):
    """Return prime factorization as dict {prime: exponent}."""
    factors = {}
    d = 2
    while d * d <= num:
        while num % d == 0:
            factors[d] = factors.get(d, 0) + 1
            num //= d
        d += 1
    if num > 1:
        factors[num] = factors.get(num, 0) + 1
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from a factorization dict."""
    divisors = [1]
    for p, e in factors.items():
        current = []
        power = 1
        for i in range(e + 1):
            for d in divisors:
                current.append(d * power)
            power *= p
        divisors = current
    return divisors

def _merge_factorizations(f1, f2):
    """Merge two factorization dicts, summing exponents."""
    merged = dict(f1)
    for p, e in f2.items():
        merged[p] = merged.get(p, 0) + e
    return merged

def _divisors_limited(factors, limit):
    """Generate divisors of number defined by factors, only those < limit."""
    divisors = [1]
    for p, e in factors.items():
        current = []
        power = 1
        for i in range(e + 1):
            for d in divisors:
                val = d * power
                if val < limit:
                    current.append(val)
            power *= p
            if power >= limit:
                break
        divisors = current
    return divisors

def construct():
    deadline = time.time() + 82  # safety margin
    results = [None] * 254
    MAX_DENOM = 10**12
    
    # Precompute factorizations for all n from 3 to 256
    factor_cache = {}
    for n in range(3, 257):
        factor_cache[n] = _factorize(n)
    
    # Precompute n^2 factorizations and their divisors
    n2_factors = {}
    n2_divisors = {}
    for n in range(3, 257):
        fn = factor_cache[n]
        fn2 = {p: 2*e for p, e in fn.items()}
        n2_factors[n] = fn2
        n2_divisors[n] = _divisors_from_factors(fn2)
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        
        found = None
        fn = factor_cache[n]
        n2_divs = n2_divisors[n]
        n2_fact = n2_factors[n]
        
        # x range: n/4 < x <= 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Try x values in an order that's likely to find solutions quickly
        # Prefer x values that share factors with n (makes a = 4x-n have structure)
        x_candidates = list(range(x_min, x_max + 1))
        
        for x in x_candidates:
            if time.time() > deadline:
                break
            
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            b2 = b * b
            
            # We need divisors d of b^2 = n^2 * x^2 with d < b
            # b^2 = n^2 * x^2, so factor of b^2 = merge(n^2 factors, x^2 factors)
            fx = factor_cache.get(x)
            if fx is None:
                fx = _factorize(x)
                factor_cache[x] = fx
            fx2 = {p: 2*e for p, e in fx.items()}
            
            # Merge n^2 and x^2 factorizations
            b2_fact = _merge_factorizations(n2_fact, fx2)
            
            # Get divisors of b^2 that are < b
            # This could be expensive, so limit it
            divs = _divisors_limited(b2_fact, b)
            
            for d in divs:
                if (b + d) % a != 0:
                    continue
                y = (b + d) // a
                if y <= x:
                    continue
                if b2 % d != 0:
                    continue
                z_val = b2 // d
                if (b + z_val) % a != 0:
                    continue
                z = (b + z_val) // a
                if z <= y:
                    continue
                if y > MAX_DENOM or z > MAX_DENOM:
                    continue
                found = [x, y, z]
                break
            if found:
                break
        
        results[idx] = found
    
    return results
