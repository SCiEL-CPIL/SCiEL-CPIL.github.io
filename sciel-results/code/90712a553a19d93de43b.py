import time
import math

def construct():
    deadline = time.time() + 85  # 85s out of 90s to be safe
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x must satisfy 4x - n > 0, so x >= n//4 + 1
        # Also x should be reasonably small to keep b = nx manageable
        # Try x from n//4+1 upward, but limit the range
        
        x_min = n // 4 + 1
        # We need distinct denominators, so let's try a range of x values
        # For each x, we factor b = n*x and enumerate divisors of b^2
        
        for x in range(x_min, x_min + 200):  # Try up to 200 values of x
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # Factor b to get divisors of b^2
            # b^2 = (n*x)^2
            # We need divisors d of b^2 such that:
            # y = (d + b) / a is a positive integer
            # z = (b^2//d + b) / a is a positive integer
            # y != x, z != x, y != z, y > 0, z > 0, y <= 10^12, z <= 10^12
            
            # Factor b
            temp = b
            factors = {}
            f = 2
            while f * f <= temp:
                while temp % f == 0:
                    factors[f] = factors.get(f, 0) + 1
                    temp //= f
                f += 1
            if temp > 1:
                factors[temp] = factors.get(temp, 0) + 1
            
            # Generate all divisors of b^2
            # b^2 has prime factorization where each exponent is doubled
            primes = list(factors.keys())
            exps = [factors[p] * 2 for p in primes]
            
            # Generate divisors recursively
            divisors = [1]
            for i, (p, e) in enumerate(zip(primes, exps)):
                new_divs = []
                for d in divisors:
                    for k in range(e + 1):
                        new_divs.append(d * (p ** k))
                divisors = new_divs
            
            # For each divisor d of b^2, check if it gives valid y, z
            b2 = b * b
            for d in divisors:
                # y = (d + b) / a
                num_y = d + b
                if num_y % a != 0:
                    continue
                y = num_y // a
                if y <= 0 or y > 10**12:
                    continue
                
                # z = (b^2/d + b) / a
                if b2 % d != 0:
                    continue
                num_z = b2 // d + b
                if num_z % a != 0:
                    continue
                z = num_z // a
                if z <= 0 or z > 10**12:
                    continue
                
                # Check distinctness: x, y, z all distinct
                if y == x or z == x or y == z:
                    continue
                
                # All positive and distinct
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
    
    return answers
