import time
import math

def construct():
    answers = []
    deadline = time.time() + 50  # Safety margin
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        
        # Strategy 1: Set x = n, then 1/y + 1/z = 3/n
        # This gives (3y - n)(3z - n) = n^2
        # Let d = 3y - n, so d divides n^2, d > n, d ≡ n (mod 3)
        # y = (d + n) / 3, z = (n^2/d + n) / 3
        n_sq = n * n
        # Find all divisors of n^2
        divisors = []
        limit = int(math.isqrt(n_sq))
        for i in range(1, limit + 1):
            if n_sq % i == 0:
                divisors.append(i)
                if i != n_sq // i:
                    divisors.append(n_sq // i)
        
        # Sort divisors to try smaller d first (gives smaller y)
        divisors.sort()
        
        for d in divisors:
            if d <= n:
                continue
            if (d + n) % 3 != 0:
                continue
            y = (d + n) // 3
            other = n_sq // d
            if (other + n) % 3 != 0:
                continue
            z = (other + n) // 3
            # Check distinctness and positivity
            if y <= 0 or z <= 0:
                continue
            if y == z or y == n or z == n:
                continue
            if y > 10**12 or z > 10**12:
                continue
            # Verify the equation
            if y * z * n == 4 * y * z + 4 * n * z + 4 * n * y:
                found = [n, y, z]
                break
        
        # Strategy 2: If strategy 1 failed, try x = n + k for small k
        if found is None:
            for k in range(1, 20):
                x = n + k
                if x > 10**12:
                    break
                # 1/x + 1/y + 1/z = 4/n
                # 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
                a = 4 * x - n
                b = n * x
                if a <= 0:
                    continue
                # (a*y - b)(a*z - b) = b^2
                # Let d = a*y - b, d divides b^2, d > 0
                # y = (d + b) / a, z = (b^2/d + b) / a
                b_sq = b * b
                # Find divisors of b^2 that are congruent to -b mod a
                # Try a few divisors
                limit_b = int(math.isqrt(b_sq))
                # To keep it fast, only check divisors up to a reasonable limit
                # Actually, let's just try d values that work
                # d must satisfy: d | b^2, (d + b) % a == 0, d > 0
                # d ≡ -b (mod a)
                # So d = a*m - b for some m, and d | b^2
                # Let's try small m values
                for m in range(1, 50):
                    d = a * m - b
                    if d <= 0:
                        continue
                    if b_sq % d != 0:
                        continue
                    y = (d + b) // a
                    other = b_sq // d
                    z = (other + b) // a
                    if y <= 0 or z <= 0:
                        continue
                    if y == z or y == x or z == x:
                        continue
                    if y > 10**12 or z > 10**12:
                        continue
                    # Verify
                    if y * z * n == 4 * y * z + 4 * n * z + 4 * n * y:
                        found = [x, y, z]
                        break
                if found:
                    break
        
        # Strategy 3: Broader search over x
        if found is None and time.time() < deadline - 5:
            # x must be > n/4 (since 3/x < 4/n - 1/x... actually 1/x < 4/n so x > n/4)
            # and 3/x > 4/n - 4/n... let's think: 1/y + 1/z < 2/y (if y < z) 
            # Actually, for given x, we need 1/y + 1/z = 4/n - 1/x = (4x - n)/(nx)
            # This must be positive, so x > n/4
            # Also, 1/y + 1/z > 1/y (since z > y), so 1/y < (4x-n)/(nx), so y > nx/(4x-n)
            # And 1/y + 1/z < 2/y, so 2/y > (4x-n)/(nx), so y < 2nx/(4x-n)
            start_x = n // 4 + 1
            # Limit the search to keep it fast
            for x in range(start_x, min(start_x + 50, n + 50)):
                if x > 10**12:
                    break
                a = 4 * x - n
                b = n * x
                if a <= 0:
                    continue
                b_sq = b * b
                # Try divisors d of b^2 where d ≡ -b (mod a)
                # d = a*m - b
                for m in range(1, 100):
                    d = a * m - b
                    if d <= 0:
                        continue
                    if b_sq % d != 0:
                        continue
                    y = (d + b) // a
                    other = b_sq // d
                    z = (other + b) // a
                    if y <= 0 or z <= 0:
                        continue
                    if y == z or y == x or z == x:
                        continue
                    if y > 10**12 or z > 10**12:
                        continue
                    if y * z * n == 4 * y * z + 4 * n * z + 4 * n * y:
                        found = [x, y, z]
                        break
                if found:
                    break
        
        answers.append(found)
    
    return answers
