import time
import math
from fractions import Fraction

def construct():
    deadline = time.time() + 80
    results = [None] * 254
    LIMIT = 10**12

    # Handle n where we can use simple identities
    # 4/n = 1/(n/2) + 1/n + 1/n is not distinct
    # 4/n = 1/(n/4) + 1/(n/4) + 1/n + 1/n - that's 4 terms
    
    # Key identity: if n is divisible by 4, 4/n = 1/(n/4) + 1/(n/2) + 1/n + 1/n
    # But we need 3 distinct terms.
    # 4/n = 1/(n/2) + 1/(3n/4) + 1/(3n/2) when n%4==0? Let's verify:
    # 1/(n/2) + 1/(3n/4) + 1/(3n/2) = 2/n + 4/(3n) + 2/(3n) = 2/n + 6/(3n) = 2/n + 2/n = 4/n. Yes!
    
    # Another: 4/n = 1/(n/2) + 1/(n/2+1) + 1/((n/2)(n/2+1)) 
    # Check: 2/n + 2/(n+2) + ... this gets complicated. Let me use the general approach.
    
    # General approach: 4/n = 1/x + 1/y + 1/z
    # Multiply by xyz: 4xyz/n = yz + xz + xy
    # 4xyz = n(xy + xz + yz)
    # For each x, we need: 4yz - n(y+z) = nx
    # 4yz - ny - nz = nx
    # (4y - n)(4z - n) = n^2 + 4nx
    # Let a = 4y - n, b = 4z - n
    # Then ab = n(n + 4x)
    # And y = (a+n)/4, z = (b+n)/4 must be positive integers with a ≡ -n (mod 4), b ≡ -n (mod 4)
    
    # So for each n and each x, we factor M = n(n+4x) and find divisors a of M
    # such that a ≡ -n (mod 4) and b = M/a also ≡ -n (mod 4)
    # and y = (a+n)/4 > x (for ordering) and z = (b+n)/4 > y (for distinctness)
    # and all ≤ LIMIT
    
    # x ranges from n/4 + 1 to ... (since 1/x < 4/n means x > n/4)
    # Also x < y < z, so 1/x > 1/y > 1/z, meaning x < y < z
    # 1/x + 1/y + 1/z = 4/n, with x < y < z
    # x must be > n/4 (since 1/x < 4/n)
    # Also 3/x > 4/n (since 1/x > 1/y > 1/z, so 3/x > 1/x+1/y+1/z = 4/n), so x < 3n/4
    # So x in (n/4, 3n/4)
    
    # Precompute for all n at once using a smarter approach:
    # For each n, try small x values first (x near n/4 gives largest y,z which may exceed limit)
    # Actually, let's just iterate efficiently.
    
    # Alternative: precompute all valid (x,y,z) triples with z <= some bound, 
    # index by the value of 1/x+1/y+1/z as a fraction, then look up 4/n.
    # But the space is too large for z up to 10^12.
    
    # Let's use the (4y-n)(4z-n) = n(n+4x) approach with factorization.
    
    def factorize(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    
    def divisors_from_factors(factors):
        if not factors:
            return [1]
        divs = [1]
        for p, e in factors.items():
            current = []
            pow_p = 1
            for k in range(e + 1):
                for d in divs:
                    current.append(d * pow_p)
                pow_p *= p
            divs = current
        return divs
    
    # Try formulas first for speed
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline - 5:
            break
        
        # Case 1: n divisible by 4
        if n % 4 == 0:
            x = n // 2
            y = 3 * n // 4
            z = 3 * n // 2
            if len(set([x, y, z])) == 3 and x <= LIMIT and y <= LIMIT and z <= LIMIT:
                # Verify: 1/x + 1/y + 1/z == 4/n
                # x = n/2, y = 3n/4, z = 3n/2
                # 1/x = 2/n, 1/y = 4/(3n), 1/z = 2/(3n)
                # sum = 2/n + 4/(3n) + 2/(3n) = 2/n + 6/(3n) = 2/n + 2/n = 4/n ✓
                if x * 3 * n == 4 * y * z and x * 3 * z == 4 * y * x and y * 3 * x == 4 * z * y:
                    # Cross-multiply check: 1/x+1/y+1/z = 4/n
                    # (yz + xz + xy)/(xyz) = 4/n
                    # n(yz + xz + xy) = 4xyz
                    if n * (y * z + x * z + x * y) == 4 * x * y * z:
                        results[idx] = [x, y, z]
                        continue
        
        # Case 2: n even but not divisible by 4 (n = 2k, k odd)
        if n % 2 == 0:
            # Try: x = n/2, then 4/n - 1/(n/2) = 4/n - 2/n = 2/n = 1/y + 1/z
            # 2/n = 1/y + 1/z => 2yz = n(y+z) => (2y-n)(2z-n) = n^2
            # So y = (a+n)/2, z = (b+n)/2 where ab = n^2
            x = n // 2
            nf = factorize(n)
            n2f = {p: 2*e for p, e in nf.items()}
            n2_divs = divisors_from_factors(n2f)
            found = None
            for a in n2_divs:
                b = (n * n) // a
                if a * b != n * n:
                    continue
                if (a + n) % 2 != 0:
                    continue
                if (b + n) % 2 != 0:
                    continue
                y = (a + n) // 2
                z = (b + n) // 2
                if y <= 0 or z <= 0:
                    continue
                if y == x or z == x or y == z:
                    continue
                if y > LIMIT or z > LIMIT:
                    continue
                # Verify
                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                    found = [x, y, z]
                    break
            if found:
                results[idx] = found
                continue
        
        # General case: iterate x from floor(n/4)+1 to ceil(3n/4)-1
        # For each x, compute M = n*(n+4x), factor it, find valid (a,b) pairs
        x_lo = n // 4 + 1
        x_hi = (3 * n + 3) // 4  # inclusive upper bound approx
        
        found = None
        for x in range(x_lo, x_hi + 1):
            if time.time() > deadline - 3:
                break
            M = n * (n + 4 * x)
            if M <= 0:
                continue
            # Need a ≡ -n (mod 4) and b ≡ -n (mod 4) where a*b = M
            nf = factorize(M)
            M_divs = divisors_from_factors(nf)
            for a in M_divs:
                if a * a > M:
                    break
                if M % a != 0:
                    continue
                b = M // a
                # Check congruences
                if (a + n) % 4 != 0:
                    continue
                if (b + n) % 4 != 0:
                    continue
                y = (a + n) // 4
                z = (b + n) // 4
                if y <= 0 or z <= 0:
                    continue
                if y == x or z == x or y == z:
                    continue
                if y > LIMIT or z > LIMIT:
                    continue
                # Verify
                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                    found = [x, y, z]
                    break
            if found:
                break
        
        # Also try swapping: a and b give y and z, but we should also check if
        # y < x < z or other orderings work. The (4y-n)(4z-n) form assumes we fix x
        # and solve for y,z. We should ensure y != z and all distinct.
        
        results[idx] = found
    
    return results
