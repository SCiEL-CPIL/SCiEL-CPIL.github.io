"""Solve 1/x+1/y+1/z=4/n for n=3..256 with distinct denominators."""
import math
import time
from collections import defaultdict

def _factorize(n):
    """Return prime factorization of n as list of (prime, exponent)."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            exp = 0
            while n % d == 0:
                n //= d
                exp += 1
            factors.append((d, exp))
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append((n, 1))
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from prime factorization."""
    divs = [1]
    for p, e in factors:
        new_divs = []
        pe = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * pe)
            pe *= p
        divs = new_divs
    return divs

def _try_closed_form(n):
    """Try known closed-form patterns for 4/n. Returns [x,y,z] or None."""
    BIG = 10**12
    
    def _valid(x, y, z):
        if x == y or y == z or x == z:
            return False
        if x <= 0 or y <= 0 or z <= 0:
            return False
        if x > BIG or y > BIG or z > BIG:
            return False
        return n * (y*z + x*z + x*y) == 4 * x * y * z
    
    # Pattern 1: n = 3m, m odd (n ≡ 3 mod 6): x=m, y=3m+1, z=3m(3m+1)
    if n % 6 == 3:
        m = n // 3
        if _valid(m, 3*m + 1, 3*m*(3*m + 1)):
            return [m, 3*m + 1, 3*m*(3*m + 1)]
    
    # Pattern 2: n = 4m, m odd (n ≡ 4 mod 8): x=m+2, y=(m+1)(m+2), z=m(m+1)
    if n % 8 == 4:
        m = n // 4
        if _valid(m + 2, (m + 1)*(m + 2), m*(m + 1)):
            return [m + 2, (m + 1)*(m + 2), m*(m + 1)]
    
    # Pattern 3: n = 5m, m odd (n ≡ 5 mod 10): x=2m, y=5m, z=10m
    if n % 10 == 5:
        m = n // 5
        if _valid(2*m, 5*m, 10*m):
            return [2*m, 5*m, 10*m]
    
    # Pattern 4: n divisible by 8: x=n/4, y=n/2, z=n
    if n % 8 == 0:
        if _valid(n // 4, n // 2, n):
            return [n // 4, n // 2, n]
    
    # Pattern 5: n divisible by 3 but not 6 (n ≡ 0 mod 3, n ≡ 1 or 2 mod 2, i.e., n odd and 3|n)
    # Actually for n divisible by 3: try x = n/3, then 1/y+1/z = 4/n - 3/n = 1/n
    # So 1/y+1/z = 1/n, giving y = n+k, z = n(n+k)/k for k|n^2
    # Try k = n: y = 2n, z = 2n -> not distinct
    # Try k = n/2 if n even... let's just handle specific cases
    
    # Pattern 6: n even, n/2 is integer. Try x = n/2, then 1/y+1/z = 4/n - 2/n = 2/n
    # 1/y+1/z = 2/n => (y+z)/(yz) = 2/n => n(y+z) = 2yz
    # Let y = n+k, then z = n(n+k)/(2k)
    # For k = n: z = n(2n)/(2n) = n, y = 2n, so {n/2, n, 2n}
    if n % 2 == 0:
        if _valid(n // 2, n, 2 * n):
            return [n // 2, n, 2 * n]
    
    # Pattern 7: Try x = n+1 generally
    # 1/y+1/z = 4/n - 1/(n+1) = (4(n+1)-n)/(n(n+1)) = (3n+4)/(n(n+1))
    # This doesn't simplify as nicely, skip for now
    
    return None

def construct():
    deadline = time.time() + 85
    answers = [None] * 254
    LIMIT = 10**12
    
    # Precompute factorizations for values we'll need
    max_x = 6000
    factor_cache = {}
    for v in range(1, max_x + 1):
        factor_cache[v] = _factorize(v)
    
    # Precompute divisors for all values up to max_x to speed up lookup
    divisor_cache = {}
    for v in range(1, max_x + 1):
        divisor_cache[v] = _divisors_from_factors(factor_cache[v])
    
    for idx in range(254):
        if time.time() > deadline:
            break
        n = idx + 3
        found = None
        
        # First try closed-form patterns
        cf = _try_closed_form(n)
        if cf is not None:
            answers[idx] = cf
            continue
        
        # Fallback: search with optimized divisor enumeration
        # Strategy: try x = n+1 first (high success rate), then expand
        x_min = n // 4 + 1
        x_max = min(4 * n + 10, max_x)
        
        # Build ordered list: n+1 first, then sequential from x_min
        seen = set()
        x_list = []
        for x in [n + 1]:
            if x_min <= x <= x_max and x not in seen:
                x_list.append(x)
                seen.add(x)
        for x in range(x_min, x_max + 1):
            if x not in seen:
                x_list.append(x)
        
        for x in x_list:
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            b_sq = b * b
            
            # Instead of enumerating all divisors of b_sq, iterate over divisors of b
            # and check both d and b_sq/d implicitly
            # Actually, we need divisors d of b_sq such that (d+b) % a == 0 and (b_sq/d+b) % a == 0
            # Key insight: if d | b_sq, then b_sq/d | b_sq too, so we can just iterate divisors of b_sq
            # But b_sq can be huge. Instead, note that d must satisfy d ≡ -b (mod a) and b_sq/d ≡ -b (mod a)
            # So d ≡ -b (mod a) and b_sq/d ≡ -b (mod a)
            # From d ≡ -b (mod a): d = a*k - b for some k
            # Then b_sq/d = b_sq/(a*k-b) must be integer and ≡ -b (mod a)
            
            # Simpler approach: iterate over divisors of b (not b_sq), since if d|b_sq and d|b, 
            # then we cover a subset. But we need all d|b_sq.
            # 
            # Better: enumerate divisors of b, and for each divisor d of b, 
            # b_sq/d = b*(b/d) is also determined. But d might not divide b.
            #
            # Most efficient: enumerate divisors of b_sq directly but use the factor cache
            # to avoid re-factorization. The issue is b_sq = (n*x)^2 can have many divisors.
            #
            # Alternative: use the fact that d must be ≡ -b (mod a).
            # So d = a*t - b for integer t >= 1.
            # And d must divide b_sq.
            # So a*t - b | b_sq, i.e., b_sq % (a*t - b) == 0.
            # Also need b_sq/(a*t-b) + b ≡ 0 (mod a), i.e., b_sq/(a*t-b) ≡ -b (mod a).
            
            # Let's try the modular approach: d = a*t - b, check if d | b_sq
            # t ranges from ceil(b/a)+1 to floor(b_sq/a)+1, which is too large.
            
            # Let's stick with divisor enumeration but optimize by only generating divisors
            # up to sqrt(b_sq) and checking pairs.
            
            # Factor b = n*x using cache
            combined = defaultdict(int)
            for p, e in factor_cache[n]:
                combined[p] += e
            for p, e in factor_cache[x]:
                combined[p] += e
            
            # Get divisors of b
            divs_b = _divisors_from_factors(list(combined.items()))
            
            # For each divisor d of b, d^2 divides b_sq, and b_sq/d^2 is also a divisor of b_sq
            # But we need all divisors of b_sq, not just squares.
            # 
            # Actually, let's just enumerate divisors of b_sq properly but with early termination.
            # The number of divisors of (n*x)^2 where n,x <= 256*4 is manageable.
            
            # Use modular constraint to reduce search space
            # d must satisfy: d ≡ -b (mod a) and b_sq/d ≡ -b (mod a)
            # So d = a*t - b for some t, and we need d | b_sq
            
            # Iterate over divisors of b_sq using factorization, but with early exit
            # Combine factorizations
            combined_sq = defaultdict(int)
            for p, e in factor_cache[n]:
                combined_sq[p] += 2 * e
            for p, e in factor_cache[x]:
                combined_sq[p] += 2 * e
            
            bf = list(combined_sq.items())
            divs = _divisors_from_factors(bf)
            
            # Filter divisors using modular constraint first
            for d in divs:
                if time.time() > deadline:
                    break
                # Check modular constraint: (d + b) % a == 0
                if (d + b) % a != 0:
                    continue
                e_val = b_sq // d
                num_z = e_val + b
                if num_z % a != 0:
                    continue
                y = (d + b) // a
                z = num_z // a
                if y <= 0 or z <= 0:
                    continue
                if y > LIMIT or z > LIMIT:
                    continue
                if x == y or x == z or y == z:
                    continue
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            if found:
                break
        
        answers[idx] = found
    
    return answers
