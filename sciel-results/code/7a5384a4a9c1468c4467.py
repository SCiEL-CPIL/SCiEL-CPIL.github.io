import time
import math

def construct():
    start = time.time()
    deadline = start + 85.0
    LIMIT = 10**12
    answers = [None] * 254
    
    # Precompute primes up to 10000 for faster factorization
    primes = []
    for i in range(2, 10001):
        is_prime = True
        for p in primes:
            if p * p > i:
                break
            if i % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
    
    def factorize(num):
        factors = {}
        temp = num
        for p in primes:
            if p * p > temp:
                break
            while temp % p == 0:
                factors[p] = factors.get(p, 0) + 1
                temp //= p
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        return factors
    
    def get_divisors_of_square(b):
        """Get all divisors of b^2 efficiently"""
        factors = factorize(b)
        divisors = [1]
        for p, e in factors.items():
            max_exp = 2 * e
            new_divs = []
            p_powers = [1]
            for k in range(1, max_exp + 1):
                p_powers.append(p_powers[-1] * p)
            for d in divisors:
                for pp in p_powers:
                    new_divs.append(d * pp)
            divisors = new_divs
        return divisors

    def find_solution(n):
        if time.time() > deadline:
            return None
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        if x_max < x_min:
            return None
            
        # Try heuristic x values first: x = n/2, x = 2n/3, x = 3n/4, etc.
        heuristic_xs = set()
        for frac_num, frac_den in [(1,2), (2,3), (3,4), (1,3), (2,5), (3,5), (1,4), (3,8), (5,8), (4,5), (5,6), (7,8)]:
            x_val = n * frac_num // frac_den
            if x_min <= x_val <= x_max:
                heuristic_xs.add(x_val)
            x_val2 = n * frac_num // frac_den + 1
            if x_min <= x_val2 <= x_max:
                heuristic_xs.add(x_val2)
            x_val3 = n * frac_num // frac_den - 1
            if x_min <= x_val3 <= x_max:
                heuristic_xs.add(x_val3)
        
        # Try heuristic values first
        for x in sorted(heuristic_xs):
            if time.time() > deadline:
                return None
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            b2 = b * b
            
            # Enumerate divisors of b^2
            divs = get_divisors_of_square(b)
            for u in divs:
                if u >= b:
                    continue
                if (u + b) % a != 0:
                    continue
                v = b2 // u
                if (v + b) % a != 0:
                    continue
                
                y = (u + b) // a
                z = (v + b) // a
                
                if y <= 0 or z <= 0:
                    continue
                if y > LIMIT or z > LIMIT:
                    continue
                if x == y or x == z or y == z:
                    continue
                
                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                    return [x, y, z]
        
        # Fall back to sequential search for remaining x values
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                return None
            if x in heuristic_xs:
                continue
                
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            b2 = b * b
            
            # Use divisor enumeration for all b values
            # But limit the number of divisors we check if b is very large
            divs = get_divisors_of_square(b)
            
            # Sort divisors to find solutions faster
            # u must be < b for y to be positive
            for u in divs:
                if u >= b:
                    continue
                if (u + b) % a != 0:
                    continue
                v = b2 // u
                if (v + b) % a != 0:
                    continue
                
                y = (u + b) // a
                z = (v + b) // a
                
                if y <= 0 or z <= 0:
                    continue
                if y > LIMIT or z > LIMIT:
                    continue
                if x == y or x == z or y == z:
                    continue
                
                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                    return [x, y, z]
        return None

    # First pass: solve for all n
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        answers[idx] = find_solution(n)
    
    # Check how many are solved
    solved = sum(1 for a in answers if a is not None)
    
    # If we have time, try to solve the remaining ones with a wider search
    if solved < 254:
        remaining = [n for idx, n in enumerate(range(3, 257)) if answers[idx] is None]
        
        for n in remaining:
            if time.time() > deadline - 2:
                break
            
            found = None
            x_min = n // 4 + 1
            x_max = (3 * n) // 4
            
            # Try more x values with direct y search
            for x in range(x_min, x_max + 1):
                if time.time() > deadline - 1:
                    break
                    
                a = 4 * x - n
                if a <= 0:
                    continue
                b = n * x
                
                # Try y values with a larger window
                y_min = b // a + 1
                y_max = (2 * b) // a
                
                if y_max < y_min:
                    continue
                    
                # Limit the search range based on available time
                search_limit = 1000 if time.time() < deadline - 5 else 200
                for y in range(y_min, min(y_max + 1, y_min + search_limit)):
                    if time.time() > deadline - 1:
                        break
                    
                    denom = a * y - b
                    if denom <= 0 or b * y % denom != 0:
                        continue
                    
                    z = b * y // denom
                    
                    if z <= 0 or z > LIMIT:
                        continue
                    if x == y or x == z or y == z:
                        continue
                    if z <= y:
                        continue
                    
                    if n * (y * z + x * z + x * y) == 4 * x * y * z:
                        found = [x, y, z]
                        break
                
                if found:
                    break
            
            if found:
                answers[n - 3] = found

    return answers
