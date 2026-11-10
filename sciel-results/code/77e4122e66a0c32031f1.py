import time

def construct():
    start = time.time()
    deadline = start + 85.0
    LIMIT = 10**12
    answers = [None] * 254
    
    # Precompute primes up to 2000
    primes = []
    for i in range(2, 2001):
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
        factors = factorize(b)
        divisors = [1]
        for p, e in factors.items():
            max_exp = 2 * e
            new_divs = []
            for d in divisors:
                val = d
                for k in range(max_exp + 1):
                    new_divs.append(val)
                    val *= p
            divisors = new_divs
        return divisors

    def find_solution(n):
        if time.time() > deadline:
            return None
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        if x_max < x_min:
            return None
            
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                return None
                
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            b2 = b * b
            
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
        return None

    # First pass: solve for all n
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        answers[idx] = find_solution(n)

    # Check how many are solved
    solved = sum(1 for a in answers if a is not None)
    
    # If we have time, try to solve the remaining ones with a different approach
    if solved < 254:
        remaining = [n for idx, n in enumerate(range(3, 257)) if answers[idx - 2] is None]
        
        for n in remaining:
            if time.time() > deadline - 2:
                break
            
            # Try a different strategy: iterate over possible y values
            # 1/y + 1/z = 4/n - 1/x
            # For fixed x, we need 1/y + 1/z = (4x - n)/(nx)
            # So (y+z)/(yz) = (4x-n)/(nx)
            # nx(y+z) = yz(4x-n)
            # yz(4x-n) - nxy - nxz = 0
            # y[z(4x-n) - nx] = nxz
            # y = nxz / [z(4x-n) - nx]
            
            # Let's try small x values again but with a different divisor approach
            found = None
            for x in range(n // 4 + 1, (3 * n) // 4 + 1):
                if time.time() > deadline - 1:
                    break
                    
                a = 4 * x - n
                if a <= 0:
                    continue
                b = n * x
                
                # Try y values
                y_min = b // a + 1
                y_max = (2 * b) // a
                
                if y_max < y_min:
                    continue
                    
                for y in range(y_min, min(y_max + 1, y_min + 200)):
                    if time.time() > deadline - 1:
                        break
                    
                    c = a * y - b
                    if c <= 0:
                        continue
                    d = b * y
                    
                    # z = d * a / (a * d - b * c) ... let me re-derive
                    # 1/y + 1/z = a/b
                    # (y+z)/(yz) = a/b
                    # b(y+z) = a*yz
                    # a*yz - by - bz = 0
                    # z(ay - b) = by
                    # z = by / (ay - b)
                    
                    denom = a * y - b
                    if denom <= 0 or b * y % denom != 0:
                        continue
                    
                    z = b * y // denom
                    
                    if z <= 0 or z > LIMIT:
                        continue
                    if x == y or x == z or y == z:
                        continue
                    if z <= y:  # Avoid duplicates
                        continue
                    
                    if n * (y * z + x * z + x * y) == 4 * x * y * z:
                        found = [x, y, z]
                        break
                
                if found:
                    break
            
            if found:
                answers[n - 3] = found

    return answers
