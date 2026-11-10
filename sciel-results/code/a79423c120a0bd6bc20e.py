import time
import math

def construct():
    start = time.time()
    deadline = start + 85.0
    LIMIT = 10**12
    answers = [None] * 254
    
    # Primes up to 224 (sqrt(50000) approx)
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
        # Factorize b
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
            
        # Generate divisors of b^2
        # If b = prod p^e, b^2 = prod p^(2e)
        # Divisors are prod p^k where 0 <= k <= 2e
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
            b2 = b * b
            
            divs = get_divisors_of_square(b)
            
            for u in divs:
                if u >= b:
                    continue
                # Check u ≡ -b (mod a)
                if (u + b) % a != 0:
                    continue
                v = b2 // u
                # Check v ≡ -b (mod a)
                if (v + b) % a != 0:
                    continue
                
                y = (u + b) // a
                z = (v + b) // a
                
                if y <= 0 or z <= 0:
                    continue
                if y > LIMIT or z > LIMIT:
                    continue
                
                # Distinctness
                if x == y or x == z or y == z:
                    continue
                
                # Verify
                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers[idx] = found
        
    return answers
