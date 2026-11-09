"""Direct construction of distinct three-term unit-fraction representations of 4/n."""
# EVOLVE-BLOCK-START
import time
import math

def _divisors_of(m):
    """Return all positive divisors of m (m > 0)."""
    if m <= 0:
        return []
    divs = []
    i = 1
    while i * i <= m:
        if m % i == 0:
            divs.append(i)
            if i != m // i:
                divs.append(m // i)
        i += 1
    return divs

def construct():
    deadline = time.time() + 85  # safety margin before the 90s kill
    answers = []
    
    # Track which n values have been solved for potential reuse
    solved_cache = {}
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        target_num = 4  # numerator of 4/n
        target_den = n
        
        # We need 1/x + 1/y + 1/z = 4/n
        # x is the smallest denominator.
        # 1/x < 4/n => x > n/4
        # 3/x >= 4/n => x <= 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # If x is not an integer or out of bounds, skip
        if x_min > x_max:
            answers.append(None)
            continue
        
        # Try up to 200 x values, starting from x_min
        x_limit = min(x_max, x_min + 199)
        
        for x in range(x_min, x_limit + 1):
            if time.time() > deadline:
                break
            if found:
                break
            
            # 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
            a = 4 * x - n
            b = n * x
            
            if a <= 0 or b <= 0:
                continue
                
            # 1/y + 1/z = a/b
            # (y+z)/yz = a/b
            # b(y+z) = ayz
            # ayz - by - bz = 0
            # (ay - b)(az - b) = b^2
            # Let u = ay - b and v = az - b. Then uv = b^2.
            # y = (u + b) / a and z = (v + b) / a.
            # We need y and z to be integers, so a | (u+b) and a | (v+b).
            # Also y > x and z > y (distinct positive denominators).
            # Since u*v = b^2, we can iterate over divisors u of b^2.
            # To keep the search space manageable, we can iterate over divisors of b^2.
            # However, b can be large. Let's try to be smarter.
            
            # Instead of iterating over all divisors of b^2 (which can be huge),
            # we can iterate over possible values of y in a bounded range.
            # From 1/y + 1/z = a/b and z > y:
            # 1/y > a/(2b) => y < 2b/a
            # Also y > x.
            # So y is in (x, 2b/a).
            
            y_low = max(x + 1, b // a + 1)
            y_high = (2 * b) // a
            
            if y_low > y_high:
                continue
                
            # To avoid iterating over a huge range of y, we can use the divisor property.
            # (ay - b)(az - b) = b^2
            # Let d = ay - b. Then d is a divisor of b^2.
            # Also d = ay - b => y = (d + b) / a.
            # So we need d + b to be divisible by a.
            # d ≡ -b (mod a).
            # Let's find all divisors d of b^2 such that d ≡ -b (mod a).
            # This is equivalent to finding divisors of b^2 in a specific congruence class.
            
            # A more efficient approach: iterate over divisors of b, not b^2.
            # Let g = gcd(a, b). Then a = g*a', b = g*b' with gcd(a', b') = 1.
            # 1/y + 1/z = a'/b'
            # (a'y - b')(a'z - b') = b'^2
            # This reduces the size of the number we need to factor.
            
            g = math.gcd(a, b)
            a_prime = a // g
            b_prime = b // g
            
            # Now we need (a_prime * y - b_prime) * (a_prime * z - b_prime) = b_prime^2
            # Let d1 = a_prime * y - b_prime and d2 = a_prime * z - b_prime.
            # d1 * d2 = b_prime^2
            # y = (d1 + b_prime) / a_prime
            # z = (d2 + b_prime) / a_prime
            
            # We need to find divisors d1 of b_prime^2 such that:
            # 1. (d1 + b_prime) is divisible by a_prime
            # 2. y = (d1 + b_prime) / a_prime > x
            # 3. z = (d2 + b_prime) / a_prime > y
            # 4. x, y, z are distinct
            
            # b_prime^2 can still be large, but b_prime = b/g.
            # Let's try to find divisors of b_prime^2.
            # If b_prime is small enough, we can enumerate all divisors.
            # Otherwise, we fall back to iterating over y in a limited range.
            
            bp2 = b_prime * b_prime
            
            # If bp2 is too large to factor quickly, use a bounded y-search.
            # For now, let's try the divisor approach if bp2 is not too huge.
            # A rough heuristic: if bp2 < 10^12, factor it.
            
            if bp2 <= 10**12:
                divs = _divisors_of(bp2)
                for d1 in divs:
                    if time.time() > deadline:
                        break
                    # Check congruence: d1 + b_prime ≡ 0 (mod a_prime)
                    if (d1 + b_prime) % a_prime != 0:
                        continue
                    
                    y = (d1 + b_prime) // a_prime
                    d2 = bp2 // d1
                    z = (d2 + b_prime) // a_prime
                    
                    # Check ordering and distinctness
                    if y <= x or z <= y:
                        continue
                    if x == y or y == z or x == z:
                        continue
                    
                    # Verify the solution
                    lhs_num = n * y * z + n * x * z + n * x * y
                    rhs_num = 4 * x * y * z
                    if lhs_num == rhs_num:
                        if x <= 10**12 and y <= 10**12 and z <= 10**12:
                            found = [x, y, z]
                            break
            else:
                # Fallback: iterate over y in a wider range
                # Use a larger budget but with early termination
                max_y_checks = 50000
                for y in range(y_low, min(y_low + max_y_checks, y_high + 1)):
                    if time.time() > deadline:
                        break
                    denom = a * y - b
                    if denom <= 0:
                        continue
                    if (b * y) % denom != 0:
                        continue
                    z = (b * y) // denom
                    if z <= y:
                        continue
                    if x == y or y == z or x == z:
                        continue
                    
                    lhs_num = n * y * z + n * x * z + n * x * y
                    rhs_num = 4 * x * y * z
                    if lhs_num == rhs_num:
                        if x <= 10**12 and y <= 10**12 and z <= 10**12:
                            found = [x, y, z]
                            break
        
        answers.append(found)
    
    return answers

# Precompute some useful values to speed up the search
def _precompute_solutions():
    """Precompute solutions for all n values using the optimized algorithm."""
    results = {}
    for n in range(3, 257):
        results[n] = None
    return results
# EVOLVE-BLOCK-END
