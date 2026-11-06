import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    answers = [None] * 254
    
    # Precompute primes for trial division
    limit = 100000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]

    def factorize(num):
        factors = {}
        temp = num
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                factors[p] = count
        if temp > 1:
            factors[temp] = 1
        return factors

    def get_divisors_of_square(factors):
        # factors is dict of p -> e for B
        # We need divisors of B^2, so exponents are 2*e
        divisors = [1]
        for p, e in factors.items():
            new_divs = []
            exp_max = 2 * e
            for d in divisors:
                val = d
                for k in range(exp_max + 1):
                    new_divs.append(val)
                    val *= p
            divisors = new_divs
        return divisors

    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
            
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        # Try x from smallest to largest? Or largest?
        # Smallest x -> largest A -> smallest B^2/A range?
        # Let's stick to x_max downwards as in original, or try x_min upwards.
        # Original used x_max downwards.
        for x in range(x_max, x_min - 1, -1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
                
            B = n * x
            
            # Find divisors U of B^2 such that U <= B
            # and (U + B) % A == 0
            # and (B^2//U + B) % A == 0
            # and resulting y, z distinct from x and each other
            
            # Factor B
            b_factors = factorize(B)
            b_sq_divs = get_divisors_of_square(b_factors)
            
            # Filter and check
            for U in b_sq_divs:
                if U > B:
                    continue
                if (U + B) % A != 0:
                    continue
                
                y = (U + B) // A
                
                V = (B * B) // U
                if (V + B) % A != 0:
                    continue
                
                z = (V + B) // A
                
                # Check distinctness and positive integers
                if y == x or z == x or y == z:
                    continue
                if y <= 0 or z <= 0:
                    continue
                if y > 1000000000000 or z > 1000000000000:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
                
        answers[idx] = found
        
    return answers
