"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    results = [None] * 254
    
    # Precompute a lookup: for small n, try to find solutions quickly
    # The key insight: for each n, we need 1/x + 1/y + 1/z = 4/n
    # Rearranging for fixed x: 1/y + 1/z = (4x - n)/(nx)
    # Let A = 4x - n, B = nx. Then (Ay - B)(Az - B) = B^2
    # So d = Ay - B must divide B^2, and y = (d+B)/A, z = (B^2/d + B)/A
    
    # For speed, we'll use a hybrid approach:
    # 1. For small x values (where A is small, B/A is large), the y-range is large
    #    but we can use the divisor structure
    # 2. For larger x values (where A is large, B/A is small), the y-range is small
    #    and direct iteration is fast
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # y must satisfy: B/A < y < 2B/A, and y > x
            y_lo = max(x + 1, B // A + 1)
            y_hi = min(2 * B // A, 10**12)
            
            if y_lo > y_hi:
                continue
            
            # For small ranges, direct iteration is fine
            # For large ranges, use divisor-based search
            range_size = y_hi - y_lo + 1
            
            if range_size < 100000:
                # Direct iteration
                for y in range(y_lo, y_hi + 1):
                    if time.time() > deadline:
                        break
                    denom = A * y - B
                    if denom <= 0:
                        continue
                    num = B * y
                    if num % denom == 0:
                        z = num // denom
                        if z > y and z <= 10**12 and x != y and y != z and x != z:
                            # Verify
                            if n * (y * z + x * z + x * y) == 4 * x * y * z:
                                found = [x, y, z]
                                break
            else:
                # Use divisor-based approach
                # We need d = A*y - B to divide B^2
                # d must satisfy: d > A*x - B (so y > x) and d < B (so y < z)
                # Also (d + B) must be divisible by A
                # Let's find divisors of B^2 that are congruent to -B mod A
                # and in the range (A*x - B, B)
                
                # Factor B to generate divisors of B^2
                B2 = B * B
                # Get divisors of B^2
                # For efficiency, we'll iterate over divisors of B and try to build up
                # Actually, let's just try small multipliers
                
                # Alternative: iterate over possible d values that divide B^2
                # d must be in (A*x - B, B) and d ≡ -B (mod A)
                # So d = k*A - B for some k, and d must divide B^2
                # d > A*x - B => k*A - B > A*x - B => k > x
                # d < B => k*A - B < B => k < 2B/A
                # So k ranges from x+1 to floor(2B/A) - 1
                # And d = k*A - B must divide B^2
                
                k_lo = x + 1
                k_hi = min(2 * B // A, 10**12 // A + 1)
                
                for k in range(k_lo, k_hi + 1):
                    if time.time() > deadline:
                        break
                    d = k * A - B
                    if d <= 0:
                        continue
                    if d >= B:
                        break
                    # Check if d divides B^2
                    if B2 % d == 0:
                        y = k
                        z = (B2 // d + B) // A
                        # Check z is integer
                        if (B2 // d + B) % A != 0:
                            continue
                        z = (B2 // d + B) // A
                        if z > y and z <= 10**12 and x != y and y != z and x != z:
                            # Verify
                            if n * (y * z + x * z + x * y) == 4 * x * y * z:
                                found = [x, y, z]
                                break
            
            if found:
                break
        
        results[idx] = found
    
    return results
# EVOLVE-BLOCK-END
