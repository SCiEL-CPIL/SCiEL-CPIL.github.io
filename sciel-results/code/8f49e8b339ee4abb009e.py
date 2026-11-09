import time
import math

def _sieve_spf(limit):
    """Compute smallest prime factor for all numbers up to limit."""
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

# Precompute SPF up to a reasonable limit.
# n is up to 256, x is up to ~3n/4 ~ 192.
# b = nx is up to 256 * 192 ~ 49152.
# b^2 is up to ~2.4 * 10^9.
# Factorizing b requires SPF up to b.
# Let's set limit to 10^6 to be safe for intermediate values if x grows, though x_max is small.
# Actually, x_max = 3n/4. For n=256, x_max = 192.
# b = n*x <= 256 * 192 = 49152.
# So SPF up to 50000 is sufficient for b.
# However, if we try larger x in future optimizations, we might need more.
# Let's stick to 10^6 for safety and speed.
_SPF_LIMIT = 10**6
_SPF = _sieve_spf(_SPF_LIMIT)

def _factorize(n):
    """Return a list of (prime, exponent) for n using SPF."""
    factors = []
    if n <= 1:
        return factors
    while n > 1:
        if n <= _SPF_LIMIT:
            p = _SPF[n]
        else:
            # Fallback for large n (should not happen often if bounds are tight)
            # Trial division for remaining part
            p = None
            for i in range(2, int(math.isqrt(n)) + 1):
                if n % i == 0:
                    p = i
                    break
            if p is None:
                p = n
        
        exp = 0
        while n % p == 0:
            n //= p
            exp += 1
        factors.append((p, exp))
    return factors

def _divisors_from_factors(factors):
    """Generate all divisors from a list of (prime, exponent)."""
    divs = [1]
    for p, e in factors:
        current_divs = []
        pe = 1
        for _ in range(e + 1):
            for d in divs:
                current_divs.append(d * pe)
            pe *= p
        divs = current_divs
    return divs

def _find_solution_for_n(n):
    """Find distinct x,y,z with 1/x+1/y+1/z=4/n, x<y<z, all <= 10^12."""
    x_min = n // 4 + 1
    x_max = (3 * n) // 4
    
    if x_max < x_min:
        return None
    
    # We iterate x from x_min to x_max.
    # For each x, we solve 1/y + 1/z = 4/n - 1/x = (4x-n)/(nx) = a/b
    # y+z = ab, yz = b^2
    # y,z are roots of t^2 - ab t + b^2 = 0
    # Discriminant D = a^2 b^2 - 4b^2 = b^2(a^2 - 4)
    # For y,z to be integers, a^2 - 4 must be a perfect square? No, that's for specific forms.
    # General method:
    # 1/y + 1/z = a/b => b(y+z) = a y z => a y z - b y - b z = 0
    # Multiply by a: a^2 y z - ab y - ab z = 0
    # Add b^2: (ay - b)(az - b) = b^2
    # Let u = ay - b, v = az - b. Then uv = b^2.
    # y = (u+b)/a, z = (v+b)/a.
    # We need u, v to be divisors of b^2 such that (u+b) and (v+b) are divisible by a.
    # Also y, z must be distinct integers > x.
    
    for x in range(x_min, x_max + 1):
        a = 4 * x - n
        b = n * x
        
        if a <= 0:
            continue
            
        g = math.gcd(a, b)
        a_red = a // g
        b_red = b // g
        
        # We need to find divisors d of b_red^2 such that:
        # y = (b_red + d) / a_red is an integer
        # z = (b_red + b_red^2/d) / a_red is an integer
        # And y > x, z > y, y != z, y,z <= 10^12
        
        b_red_sq = b_red * b_red
        
        factors_b = _factorize(b_red)
        factors_sq = [(p, 2 * e) for p, e in factors_b]
        
        divs = _divisors_from_factors(factors_sq)
        
        for d in divs:
            num_y = b_red + d
            if num_y % a_red != 0:
                continue
            y = num_y // a_red
            
            if y <= x:
                continue
            
            num_z = b_red + b_red_sq // d
            if num_z % a_red != 0:
                continue
            z = num_z // a_red
            
            if z <= y:
                continue
            
            if y > 10**12 or z > 10**12:
                continue
            
            # Verify via cross multiplication
            # 1/x + 1/y + 1/z = 4/n
            # n(yz + xz + xy) = 4xyz
            lhs_num = n * (y * z + x * z + x * y)
            rhs_num = 4 * x * y * z
            if lhs_num == rhs_num:
                return [x, y, z]
    
    return None

def construct():
    deadline = time.time() + 85
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = _find_solution_for_n(n)
        answers.append(found)
    
    return answers
