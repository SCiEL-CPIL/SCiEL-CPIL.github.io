"""Solve 1/x+1/y+1/z=4/n for n=3..256 with distinct denominators."""
import math
import time

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

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    answers = [None] * 254  # index 0 -> n=3, index 253 -> n=256
    LIMIT = 10**12
    
    for idx in range(254):
        if time.time() > deadline:
            break
        n = idx + 3
        a_target = 4 * n
        # x must satisfy 4x > n, so x >= n//4 + 1
        # Also for the equation to have positive solutions, we need a=4x-n > 0
        # Start x from max(1, n//4 + 1), but also x should be reasonable
        # For 1/x + 1/y + 1/z = 4/n, we need x < 4n/3 (since 1/x < 4/n means x > n/4,
        # and for distinct positive, x can be up to about 4n/3 - something)
        # Actually 1/x < 4/n => x > n/4. And 1/x > 4/n - 2/(x+1) roughly...
        # Let's just iterate x from n//4+1 upward, bounded by time.
        
        found = None
        # Upper bound for x: since 1/y + 1/z <= 2/(x+1) (if y,z > x),
        # we need 4/n - 1/x <= 2/(x+1), but let's just be generous.
        # A safe upper bound: x < 4n (since 1/x < 4/n requires x > n/4, and x can't be too large)
        # Actually x can be quite large. Let's cap at 4*n to keep it manageable.
        x_max = min(4 * n + 10, 2000)  # reasonable cap
        
        for x in range(n // 4 + 1, x_max + 1):
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            b_sq = b * b
            
            # Factor b_sq
            bf = _factorize(b_sq)
            divs = _divisors_from_factors(bf)
            
            for d in divs:
                if time.time() > deadline:
                    break
                if b_sq % d != 0:
                    continue
                e = b_sq // d
                # y = (d + b) / a, z = (e + b) / a
                num_y = d + b
                num_z = e + b
                if num_y % a != 0 or num_z % a != 0:
                    continue
                y = num_y // a
                z = num_z // a
                if y <= 0 or z <= 0:
                    continue
                if y > LIMIT or z > LIMIT:
                    continue
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                # Verify: 1/x + 1/y + 1/z == 4/n
                # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
                lhs = n * (y * z + x * z + x * y)
                rhs = 4 * x * y * z
                if lhs == rhs:
                    found = [x, y, z]
                    break
            if found:
                break
        
        answers[idx] = found
    
    return answers
