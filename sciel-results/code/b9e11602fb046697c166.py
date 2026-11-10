import time
import math

def construct():
    MAX_DENOM = 10**12
    DEADLINE = time.time() + 85.0  # safety margin before 90s kill
    answers = []
    
    def get_divisors(num):
        """Get all divisors of num using trial division factorization."""
        if num == 0:
            return [0]
        factors = {}
        d = 2
        temp = num
        while d * d <= temp:
            while temp % d == 0:
                factors[d] = factors.get(d, 0) + 1
                temp //= d
            d += 1
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        
        # Generate divisors from prime factorization
        divisors = [1]
        for prime, exp in factors.items():
            new_divs = []
            for d in divisors:
                for e in range(exp + 1):
                    new_divs.append(d * (prime ** e))
            divisors = new_divs
        return divisors
    
    def find_solution(n):
        """Find distinct positive integer denominators x,y,z with 1/x+1/y+1/z=4/n."""
        # We want 1/x + 1/y + 1/z = 4/n
        # WLOG assume x <= y <= z
        # Then 1/x >= 4/(3n), so x <= 3n/4
        # And 1/x < 4/n, so x > n/4
        # So n/4 < x <= 3n/4
        # 
        # Given x, we need 1/y + 1/z = 4/n - 1/x = (4x-n)/(nx)
        # Let a = 4x-n, b = nx
        # Then 1/y + 1/z = a/b
        # So (y+z)/(yz) = a/b
        # b(y+z) = a*yz
        # a*yz - b*y - b*z = 0
        # a*yz - b*y - b*z + b^2/a = b^2/a (if a divides b^2)
        # (ay-b)(az-b) = b^2
        # 
        # So we need to find divisors d of b^2 such that:
        # ay - b = d, az - b = b^2/d
        # y = (d+b)/a, z = (b^2/d + b)/a
        # Both must be positive integers, y >= x, z >= y
        # And x, y, z must all be distinct
        
        for x in range(n // 4 + 1, 3 * n // 4 + 1):
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            b_sq = b * b
            
            # Get divisors of b^2
            divs = get_divisors(b_sq)
            
            for d in divs:
                # y = (d + b) / a must be integer
                if (d + b) % a != 0:
                    continue
                y = (d + b) // a
                if y < x:  # We want y >= x for ordering
                    continue
                if y > MAX_DENOM:
                    continue
                
                rem = b_sq // d
                # z = (rem + b) / a must be integer
                if (rem + b) % a != 0:
                    continue
                z = (rem + b) // a
                if z < y:  # We want z >= y for ordering
                    continue
                if z > MAX_DENOM:
                    continue
                
                # Check distinctness: x, y, z all distinct
                if x != y and x != z and y != z:
                    # Verify the equation
                    if x * y * z * 4 == n * (y * z + x * z + x * y):
                        return [x, y, z]
        
        # If the standard range didn't work, try a broader search
        # x can be larger than 3n/4 in some cases
        for x in range(3 * n // 4 + 1, 2 * n + 1):
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            b_sq = b * b
            
            # Only check a limited number of divisors for performance
            divs = get_divisors(b_sq)
            if len(divs) > 10000:
                # Too many divisors, skip this x
                continue
                
            for d in divs:
                if (d + b) % a != 0:
                    continue
                y = (d + b) // a
                if y < x:
                    continue
                if y > MAX_DENOM:
                    continue
                
                rem = b_sq // d
                if (rem + b) % a != 0:
                    continue
                z = (rem + b) // a
                if z < y:
                    continue
                if z > MAX_DENOM:
                    continue
                
                if x != y and x != z and y != z:
                    if x * y * z * 4 == n * (y * z + x * z + x * y):
                        return [x, y, z]
        
        return None
    
    for n in range(3, 257):
        if time.time() > DEADLINE:
            answers.append(None)
            continue
        
        found = find_solution(n)
        answers.append(found)
    
    return answers
