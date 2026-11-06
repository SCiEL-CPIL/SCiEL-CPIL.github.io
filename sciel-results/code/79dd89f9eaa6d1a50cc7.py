import time
from math import isqrt
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    answers = [None] * 254
    LIMIT = 10**12
    
    # Primes up to 300 for factoring B
    # B = n*x where n<=256, x<=3n/4, so B <= 256*192 = 49152.
    # sqrt(B) <= 222. Primes up to 250 are sufficient.
    primes = []
    is_prime = [True] * 300
    is_prime[0] = is_prime[1] = False
    for i in range(2, 300):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, 300, i):
                is_prime[j] = False

    # Cache for factorizations and divisors
    factor_cache = {}
    divisors_cache = {}

    def factorize(n):
        if n in factor_cache:
            return factor_cache[n]
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
        factor_cache[n] = factors
        return factors

    def get_divisors_squared(factors):
        key = tuple(sorted(factors.items()))
        if key in divisors_cache:
            return divisors_cache[key]
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
        divs.sort()  # Sort so smaller U comes first
        divisors_cache[key] = divs
        return divs

    @lru_cache(maxsize=None)
    def find_solution(n):
        # Returns tuple (x, y, z) or None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            A = 4 * x - n
            if A <= 0:
                continue
                
            B = n * x
            
            # Early skip: y = (U+B)/A >= (1+B)/A. For y > x, need (1+B)/A > x
            # i.e., 1+B > A*x = (4x-n)*x = 4x^2 - nx
            # B = nx, so 1 + nx > 4x^2 - nx => 1 + 2nx > 4x^2
            # If this fails, no solution for this x
            if 1 + 2 * n * x <= 4 * x * x:
                continue
            
            # Tighter lower bound for U: need y = (U+B)/A > x => U > A*x - B = 4x^2 - 2nx
            U_min = 4 * x * x - 2 * n * x + 1
            if U_min < 1:
                U_min = 1
            if U_min >= B:
                continue
            
            # Find U such that U | B^2 and (U + B) % A == 0
            # y = (U + B) / A
            # z = (B^2/U + B) / A
            # y > x, z > y, distinct
            
            # Get divisors of B^2 (cached)
            factors_B = factorize(B)
            divs_B2 = get_divisors_squared(factors_B)
            
            # To ensure y < z, we can assume U <= B
            # If U < B, then B^2/U > B, so z > y.
            
            for U in divs_B2:
                if U < U_min:
                    continue
                if U > B:
                    break  # Sorted, so all remaining are > B
                
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
                # B^2/U is integer since U | B^2
                z_num = (B * B) // U + B
                if z_num % A != 0:
                    continue
                z = z_num // A
                
                if z <= y:
                    continue
                if z > LIMIT:
                    continue
                
                # Distinct: x < y < z implies x!=y, y!=z, x!=z
                return (x, y, z)
        
        return None

    # Process n values. We can try to prioritize ones that might be harder or just do them in order.
    # Since we have a deadline, we should probably just iterate 3..256.
    # If we run out of time, we return what we have.
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
            
        found_tuple = find_solution(n)
        
        if found_tuple is not None:
            answers[idx] = list(found_tuple)
        # else answers[idx] remains None
        
    return answers
