import time
import math

def construct():
    start = time.time()
    deadline = start + 50.0
    LIMIT = 10**12
    answers = [None] * 254
    
    # Primes up to 224 for factorization
    primes = []
    for i in range(2, 225):
        is_prime = True
        for p in primes:
            if p * p > i:
                break
            if i % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
            
    def get_divisors_of_square(b):
        factors = {}
        temp_b = b
        for p in primes:
            if p * p > temp_b:
                break
            while temp_b % p == 0:
                factors[p] = factors.get(p, 0) + 1
                temp_b //= p
        if temp_b > 1:
            factors[temp_b] = factors.get(temp_b, 0) + 1
            
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

    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        if x_max < x_min:
            continue
            
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # Estimate y range width
            # y > b/a and y < 2b/a
            y_low = b // a + 1
            y_high = 2 * b // a
            
            if y_high < y_low:
                continue
                
            width = y_high - y_low
            
            if width < 2000:
                # Direct search
                for y in range(y_low, y_high + 1):
                    if y == x:
                        continue
                    denom = a * y - b
                    if denom <= 0:
                        continue
                    num = b * y
                    if num % denom == 0:
                        z = num // denom
                        if z <= 0 or z > LIMIT:
                            continue
                        if x == y or x == z or y == z:
                            continue
                        # Verify
                        if n * (y * z + x * z + x * y) == 4 * x * y * z:
                            found = [x, y, z]
                            break
            else:
                # Divisor search
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
                        found = [x, y, z]
                        break
            
            if found:
                break
        
        answers[idx] = found
        
    return answers
