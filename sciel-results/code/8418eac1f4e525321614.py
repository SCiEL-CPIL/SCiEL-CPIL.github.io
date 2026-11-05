"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    answers = [None] * 254  # Index 0 corresponds to n=3, index 253 to n=256
    
    # Precompute primes up to a limit sufficient for factorizing B = n*x
    # Max n = 256, max x < n, so max B = 256*255 = 65280
    # Sieve up to 100,000 to be safe.
    limit = 100000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    
    def factorize(num):
        if num <= 1:
            return []
        factors = []
        temp = num
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                factors.append((p, count))
        if temp > 1:
            factors.append((temp, 1))
        return factors

    def get_divisors_from_factorization(factors):
        divisors = [1]
        for p, exp in factors:
            new_divs = []
            p_power = 1
            for _ in range(exp + 1):
                for d in divisors:
                    new_divs.append(d * p_power)
                p_power *= p
            divisors = new_divs
        return divisors

    # Cache for divisors of B^2, keyed by B
    # B = n*x, max B = 256*255 = 65280
    divisors_cache = {}
    
    def get_divisors_of_b_squared(B):
        if B in divisors_cache:
            return divisors_cache[B]
        b_factors = factorize(B)
        b_sq_factors = [(p, 2*e) for p, e in b_factors]
        divs = get_divisors_from_factorization(b_sq_factors)
        divs.sort()
        divisors_cache[B] = divs
        return divs

    for idx in range(254):
        n = idx + 3
        
        # Check time
        if time.time() > deadline:
            break
            
        found = None
        
        # We want 1/x + 1/y + 1/z = 4/n with x < y < z
        # Rearranging: (Ay - B)(Az - B) = B^2 where A = 4x-n, B = nx
        # Let u = Ay - B, v = Az - B, uv = B^2
        # y = (u+B)/A, z = (v+B)/A must be integers with x < y < z
        
        x_start = n // 4 + 1
        
        for x in range(x_start, n):
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
                
            divisors = get_divisors_of_b_squared(B)
            
            # u must satisfy: u | B^2, u <= B, (u+B) % A == 0, (B^2/u + B) % A == 0
            # And x < (u+B)/A < (B^2/u + B)/A
            
            for u in divisors:
                if u > B:
                    break
                # Check (u + B) divisible by A
                num_y = u + B
                if num_y % A != 0:
                    continue
                y = num_y // A
                # y must be > x
                if y <= x:
                    continue
                v = (B * B) // u
                num_z = v + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                # z must be > y
                if z <= y:
                    continue
                found = [x, y, z]
                break
            
            if found:
                break
                
        answers[idx] = found
        
    return answers
# EVOLVE-BLOCK-END
