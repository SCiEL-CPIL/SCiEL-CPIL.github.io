import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    answers = [None] * 254
    LIMIT = 10**12
    
    # Primes up to 300 for factoring B
    # B <= 256 * 192 = 49152. sqrt(B) <= 222.
    # Primes up to 250 are sufficient.
    primes = []
    is_prime = [True] * 300
    is_prime[0] = is_prime[1] = False
    for i in range(2, 300):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, 300, i):
                is_prime[j] = False

    def factorize(n):
        factors = {}
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                cnt = 0
                while temp % p == 0:
                    temp //= p
                    cnt += 1
                factors[p] = cnt
        if temp > 1:
            factors[temp] = 1
        return factors

    def get_divisors_squared(factors):
        # factors is dict {prime: exp} for B
        # We need divisors of B^2, so exponents are 0..2*exp
        divs = [1]
        for p, e in factors.items():
            max_e = 2 * e
            new_divs = []
            for d in divs:
                p_pow = 1
                for i in range(max_e + 1):
                    new_divs.append(d * p_pow)
                    p_pow *= p
            divs = new_divs
        return divs

    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
            
        found = None
        
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
                
            B = n * x
            
            # Find U such that U | B^2 and (U + B) % A == 0
            # y = (U + B) / A
            # z = (B^2/U + B) / A
            # y > x, z > y, distinct
            
            # Get divisors of B^2
            factors_B = factorize(B)
            divs_B2 = get_divisors_squared(factors_B)
            
            # To ensure y < z, we can assume U <= sqrt(B^2) = B?
            # If U < B, then B^2/U > B.
            # y = (U+B)/A, z = (B^2/U + B)/A.
            # If U < B, then B^2/U > B, so z > y.
            # So we only need to check U <= B.
            
            for U in divs_B2:
                if U > B:
                    continue
                if (U + B) % A != 0:
                    continue
                
                y_num = U + B
                y = y_num // A
                
                # Check distinctness and bounds
                if y <= x:
                    continue
                if y > LIMIT:
                    continue
                
                # Check z
                # z = (B^2/U + B) / A
                # B^2/U is integer
                z_num = (B * B) // U + B
                if z_num % A != 0:
                    continue
                z = z_num // A
                
                if z <= y:
                    continue
                if z > LIMIT:
                    continue
                
                # Distinct: x < y < z implies x!=y, y!=z, x!=z
                found = [x, y, z]
                break
            
            if found:
                break
                
        answers[idx] = found
        
    return answers
