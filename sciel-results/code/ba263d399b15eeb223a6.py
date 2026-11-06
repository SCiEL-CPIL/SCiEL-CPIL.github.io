import time
import math

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Sieve for primes up to 500
    limit = 500
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    def factorize(num):
        """Returns dict of prime factors and exponents for num."""
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

    def get_divisors_from_factors(factors):
        """Generates all divisors from a prime factorization dict."""
        divisors = [1]
        for p, exp in factors.items():
            new_divs = []
            p_power = 1
            for _ in range(exp + 1):
                for d in divisors:
                    new_divs.append(d * p_power)
                p_power *= p
            divisors = new_divs
        return divisors

    answers = []
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
            
        found = None
        # x must satisfy 1/x < 4/n => x > n/4
        # and 1/x > 4/(3n) => x < 3n/4 (since x is smallest, 1/x is largest term,
        # so 1/x > 1/3 * 4/n)
        min_x = n // 4 + 1
        max_x = (3 * n) // 4 - 1 # conservative upper bound, x < 3n/4
        
        for x in range(min_x, max_x + 1):
            A = 4 * x - n
            B = n * x
            B2 = B * B
            
            # We need u * v = B2
            # u = A*y - B
            # v = A*z - B
            # y = (u+B)/A, z = (v+B)/A
            # y, z integers => u ≡ -B (mod A), v ≡ -B (mod A)
            # y < z => u < v => u < B
            # y > x => (u+B)/A > x => u+B > Ax => u > Ax - B = x(4x-n) - nx = 4x^2 - 2nx
            
            lower_bound_u = 4 * x * x - 2 * n * x
            if lower_bound_u < 0:
                lower_bound_u = 0
                
            # Get divisors of B2
            # Factor B
            b_factors = factorize(B)
            # Factor B2: exponents are doubled
            b2_factors = {p: 2 * e for p, e in b_factors.items()}
            
            divisors = get_divisors_from_factors(b2_factors)
            
            for u in divisors:
                if u >= B:
                    continue
                if u <= lower_bound_u:
                    continue
                if u != 0 and (u + B) % A != 0:
                    continue
                
                v = B2 // u
                if v <= u: # Ensure y < z
                    continue
                if (v + B) % A != 0:
                    continue
                
                y = (u + B) // A
                z = (v + B) // A
                
                # Check distinctness and bounds
                if y > x and z > y and z <= 1000000000000:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
        
    return answers
