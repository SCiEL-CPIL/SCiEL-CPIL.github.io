"""Combined program for distinct three-term unit-fraction reps of 4/n."""
# EVOLVE-BLOCK-START
import time
import math

def _divisors(n):
    """Return all divisors of n."""
    if n <= 0:
        return []
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    return divs

def construct():
    deadline = time.time() + 85.0  # Safety margin under 90s kill
    results = [None] * 254
    BOUND = 10**12
    
    # Iterate over n from 3 to 256
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
            
        found = None
        
        # x must satisfy: n/4 < x < 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n - 1) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
                
            B = n * x
            
            # We need 1/y + 1/z = A/B
            # => z = B*y / (A*y - B)
            # For z integer: A*y - B | B*y
            # Let d = A*y - B > 0, then y = (d+B)/A must be integer
            # And d | B*y = B*(d+B)/A, so d*A | B*(d+B)
            # Equivalent: d | B^2 (since d | B*d + B^2 iff d | B^2)
            
            # So iterate over divisors d of B^2 such that:
            # 1. d > 0
            # 2. (d + B) % A == 0 (so y is integer)
            # 3. y = (d+B)/A > x
            # 4. z = B*y/d > y (ensures ordering x < y < z)
            # 5. z <= BOUND
            
            B2 = B * B
            divs = _divisors(B2)
            
            for d in divs:
                if d <= 0:
                    continue
                # Check (d + B) divisible by A
                if (d + B) % A != 0:
                    continue
                y = (d + B) // A
                # Check y > x
                if y <= x:
                    continue
                # Compute z
                z = B * y // d
                # Verify: d = A*y - B
                if A * y - B != d:
                    continue
                # Check z > y and bounds
                if z <= y:
                    continue
                if z > BOUND:
                    continue
                # Check distinctness (already have x < y < z so all distinct)
                found = [x, y, z]
                break
            
            if found:
                break
                
        results[idx] = found
        
    return results
# EVOLVE-BLOCK-END
