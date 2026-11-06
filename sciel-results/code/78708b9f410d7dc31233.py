import time
from functools import lru_cache

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254
    BOUND = 10**12
    
    # Precompute small primes for faster factorization
    primes = []
    for i in range(2, 1000):
        is_p = True
        for j in primes:
            if j * j > i:
                break
            if i % j == 0:
                is_p = False
                break
        if is_p:
            primes.append(i)
    
    def factorize(n):
        factors = {}
        temp = n
        for p in primes:
            if p * p > temp:
                break
            while temp % p == 0:
                factors[p] = factors.get(p, 0) + 1
                temp //= p
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        return factors
    
    def generate_divisors_from_factors(factors):
        divs = [1]
        for p, e in factors.items():
            current_divs = []
            for d in divs:
                val = d
                for _ in range(e + 1):
                    current_divs.append(val)
                    val *= p
            divs = current_divs
        return divs
    
    # Cache for divisors of B^2
    # B = n*x, where x is in [n//4+1, 5n+10]. Max B is roughly 256 * (5*256) = 327680.
    # Actually, let's just use a dict cache.
    divisor_cache = {}
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
            
        found = None
        x_min = n // 4 + 1
        
        # We iterate x from x_min upwards.
        # The condition for y, z to be integers is that (u+B) % A == 0 and (v+B) % A == 0
        # where A = 4x - n, B = nx, u*v = B^2.
        
        for x in range(x_min, 5 * n + 10):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
                
            B = n * x
            B_sq = B * B
            
            # Get divisors of B^2
            if B in divisor_cache:
                divisors = divisor_cache[B]
            else:
                fac_B = factorize(B)
                fac_B_sq = {p: 2*e for p, e in fac_B.items()}
                divisors = generate_divisors_from_factors(fac_B_sq)
                divisor_cache[B] = divisors
            
            # We need u <= B to avoid duplicates (since u*v=B^2, if u>B then v<B)
            # And we want to find u such that (u+B) % A == 0.
            # This is equivalent to u % A == (-B) % A.
            # Let rem = (-B) % A.
            rem = (-B) % A
            
            # We can filter divisors by u % A == rem.
            # Since A can be large, this is still O(|divisors|).
            # But maybe we can generate u directly?
            # u must be a divisor of B^2.
            
            # Optimization:
            # Iterate u in divisors.
            # To speed up, we can sort divisors? No, we just need any valid pair.
            # Let's try to find u quickly.
            
            for u in divisors:
                if u > B:
                    continue
                if u % A != rem:
                    continue
                
                v = B_sq // u
                # Check v % A == rem as well?
                # (v+B) % A == (v % A + B % A) % A
                # We know v = B^2/u.
                # If u % A == rem, does v % A == rem?
                # Not necessarily. We must check.
                if v % A != rem:
                    continue
                
                y = (u + B) // A
                z = (v + B) // A
                
                if x == y or x == z or y == z:
                    continue
                if y > BOUND or z > BOUND:
                    continue
                if y < 1 or z < 1:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers[idx] = found
        
    return answers
