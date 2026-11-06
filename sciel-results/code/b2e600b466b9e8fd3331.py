"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    deadline = time.time() + 85
    results = [None] * 254
    BOUND = 10**12
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        
        x_lo = n // 4 + 1
        x_hi = (3 * n + 3) // 4
        
        for x in range(x_lo, x_hi + 1):
            a = 4 * x - n
            b = n * x
            if a <= 0:
                continue
            
            # (ay-b)(az-b) = b^2
            # U = ay-b, V = az-b, UV = b^2
            # y = (U+b)/a, z = (V+b)/a
            # Need: a | (U+b), a | (V+b), y > x, z > y, all distinct, <= BOUND
            
            b_sq = b * b
            # Iterate over divisors U of b_sq where U < b_sq/U (to avoid duplicates)
            # U must satisfy U < V, i.e., U < b_sq/U, i.e., U^2 < b_sq, i.e., U < b
            # Also y = (U+b)/a > x → U+b > ax → U > ax-b = a*x-b = a*x - n*x = x(a-n) = x(4x-n-n) = x(4x-2n)
            # And z = (V+b)/a > y → V+b > U+b → V > U (already ensured)
            # Also y > x: (U+b)/a > x → U+b > ax → U > ax - b
            # ax - b = ax - nx = x(a-n) = x(4x-n-n) = x(4x-2n) = 2x(2x-n)
            
            min_U = 2 * x * (2 * x - n) + 1  # U must be > ax - b
            if min_U < 1:
                min_U = 1
            
            # Find divisors U of b_sq in range [min_U, b)
            # For each, check a | (U+b) and a | (V+b) where V = b_sq//U
            
            found_for_x = None
            # We need U to be a divisor of b_sq
            # Iterate U from min_U to b-1, but only check divisors
            # Better: factor b_sq and generate divisors
            
            # Actually, let's just iterate U in a smart way
            # U must divide b_sq, U >= min_U, U < b
            # And (U+b) % a == 0 and (b_sq//U + b) % a == 0
            
            # Let's iterate over possible U values that are divisors
            # Generate all divisors of b_sq up to b
            sqrt_b_sq = int(math.isqrt(b_sq))
            
            # This might be too slow if b_sq is large. Let me check max b_sq.
            # b = n*x, max n=256, max x≈192, so b≈49152, b_sq≈2.4e9
            # sqrt(b_sq) ≈ 49152
            # Iterating to sqrt and checking divisors is O(sqrt(b_sq)) ≈ 50k per x
            # With ~64 x values, that's ~3.2M per n, times 254 n's = 800M. Too slow.
            
            # Better approach: for each x, just iterate U in the range and check divisibility
            # But the range [min_U, b) can be large.
            
            # Alternative: note that U ≡ -b (mod a) and U | b_sq
            # So U = k*a - b for some k, and U | b_sq
            # k*a - b | b_sq
            # Let's iterate k such that U = k*a - b is in [min_U, b) and U > 0
            # k*a - b >= min_U → k >= (min_U + b)/a
            # k*a - b < b → k < 2b/a
            # k*a - b > 0 → k > b/a
            
            k_lo = (min_U + b + a - 1) // a  # ceil((min_U+b)/a)
            k_hi = (2 * b - 1) // a  # floor((2b-1)/a), so k*a-b < b
            if k_hi < k_lo:
                continue
            
            for k in range(k_lo, k_hi + 1):
                U = k * a - b
                if U <= 0:
                    continue
                if b_sq % U != 0:
                    continue
                V = b_sq // U
                # Check y and z are integers
                if (U + b) % a != 0:
                    continue
                if (V + b) % a != 0:
                    continue
                y = (U + b) // a
                z = (V + b) // a
                # Check distinctness and bounds
                if y <= x or z <= y:
                    continue
                if y > BOUND or z > BOUND:
                    continue
                if x == y or y == z or x == z:
                    continue
                found_for_x = [x, y, z]
                break
            
            if found_for_x:
                found = found_for_x
                break
        
        results[idx] = found
    
    return results
# EVOLVE-BLOCK-END
