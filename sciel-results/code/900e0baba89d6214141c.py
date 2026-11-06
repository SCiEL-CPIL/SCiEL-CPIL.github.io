import time
import math

def construct():
    deadline = time.time() + 85
    results = [None] * 254
    BOUND = 10**12
    
    def get_divisors(n):
        """Return all divisors of n."""
        divs = set()
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return divs

    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        
        # x range: 1/x + 1/y + 1/z = 4/n
        # 3/x >= 4/n => x <= 3n/4
        # 3/z <= 4/n => z >= 3n/4, and z >= y >= x
        # Actually, WLOG x <= y <= z.
        # 3/x >= 4/n => x <= 3n/4
        # 1/x < 4/n => x > n/4
        # So x in (n/4, 3n/4]
        
        x_lo = n // 4 + 1
        x_hi = (3 * n + 3) // 4  # ceil(3n/4) roughly
        
        for x in range(x_lo, x_hi + 1):
            if time.time() > deadline:
                break
            
            a = 4 * x - n
            b = n * x
            if a <= 0:
                continue
            
            # We need (ay-b)(az-b) = b^2
            # Let u = ay-b, v = az-b. uv = b^2.
            # u < b (since y < z implies u < v, and uv=b^2 => u<b)
            # u must be a divisor of b^2.
            # Also u = ay - b => u + b = ay => u + b must be divisible by a.
            # Similarly v + b must be divisible by a.
            
            # Optimization: u must satisfy u ≡ -b (mod a)
            # Let u = k*a - b.
            # Since u > 0, k*a > b.
            # Since u < b, k*a < 2b.
            # So k is in range (b/a, 2b/a).
            
            # However, iterating k might be faster than iterating divisors if a is small?
            # b/a = n*x / (4x-n).
            # For x close to n/4, 4x-n is small, b/a is large.
            # For x close to 3n/4, 4x-n is large, b/a is small.
            
            # Let's stick to divisors of b^2, but filter by modulus.
            # Generating divisors of b^2 for b up to ~16000 (n=256, x=64) is fast.
            # Max b: n=256, x=192 (3n/4). b = 256*192 = 49152. b^2 ~ 2.4e9.
            # sqrt(2.4e9) ~ 49000.
            # Iterating to 49000 to find divisors is fine.
            
            b_sq = b * b
            # We only need divisors u < b.
            # And u ≡ -b (mod a).
            
            # Let's iterate divisors d of b_sq such that d < b.
            # We can generate them by iterating i up to sqrt(b_sq) = b.
            # For each i dividing b_sq, we get i and b_sq//i.
            # One is <= b, the other >= b.
            # We want the one < b.
            
            # Actually, if i < b, then b_sq//i > b.
            # So we just check i < b and i | b_sq.
            
            found_for_x = False
            # To speed up, we can iterate u in steps of a?
            # u = -b (mod a).
            # u_min = 1.
            # u_max = b - 1.
            # u = k*a - b.
            # 1 <= k*a - b <= b-1
            # b+1 <= k*a <= 2b-1
            # k_min = ceil((b+1)/a)
            # k_max = floor((2b-1)/a)
            
            k_min = (b + 1 + a - 1) // a  # ceil((b+1)/a)
            k_max = (2 * b - 1) // a      # floor((2b-1)/a)
            
            for k in range(k_min, k_max + 1):
                u = k * a - b
                # Check if u divides b_sq
                if b_sq % u != 0:
                    continue
                
                v = b_sq // u
                # Check if v + b is divisible by a
                # v = b_sq/u.
                # We need (v+b) % a == 0.
                # Note: u ≡ -b (mod a) is satisfied by construction.
                # Does u | b_sq and u ≡ -b (mod a) imply v ≡ -b (mod a)?
                # uv = b_sq.
                # u ≡ -b (mod a) => u = ma - b.
                # This doesn't automatically imply v ≡ -b (mod a).
                # We must check.
                
                if (v + b) % a != 0:
                    continue
                
                y = (u + b) // a
                z = (v + b) // a
                
                # Check bounds
                if y > BOUND or z > BOUND:
                    continue
                
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                
                # Check y < z (implied by u < v < b? No, u < b, v > b. So u < v. Thus y < z.)
                # Wait, if u < b, then v = b^2/u > b.
                # So u < v always.
                # So y < z always.
                
                found = [x, y, z]
                found_for_x = True
                break
            
            if found_for_x:
                break
        
        results[idx] = found
    
    return results
