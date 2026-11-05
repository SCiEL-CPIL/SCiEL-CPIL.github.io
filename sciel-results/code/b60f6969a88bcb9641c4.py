import time
import math
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    answers = [None] * 254  # Index 0 corresponds to n=3, index 253 to n=256
    
    # Precompute smallest prime factor (SPF) for fast factorization
    MAX_B = 256 * 256  # Max B = n*x < 256*256 = 65536
    spf = list(range(MAX_B + 1))
    for i in range(2, int(math.isqrt(MAX_B)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_B + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(num):
        """Fast factorization using SPF table."""
        if num <= 1:
            return []
        factors = []
        temp = num
        while temp > 1:
            p = spf[temp]
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            factors.append((p, count))
        return factors
    
    def get_divisors_from_factorization(factors):
        """Generate all divisors from prime factorization."""
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
    
    # Cache for factorizations of B values
    factor_cache = {}
    
    def get_factorization(B):
        if B in factor_cache:
            return factor_cache[B]
        factors = factorize(B)
        factor_cache[B] = factors
        return factors
    
    for idx in range(254):
        n = idx + 3
        
        # Check time
        if time.time() > deadline:
            break
            
        found = None
        
        # x must satisfy: n/4 < x < 3n/4 (from 1/x > 4/(3n) and 1/x < 4/n)
        # More precisely: 3/x > 4/n => x < 3n/4
        # And 1/x < 4/n => x > n/4
        
        x_start = n // 4 + 1
        x_end = (3 * n) // 4
        
        # Try x values - start from middle to increase chance of finding solutions faster
        # Actually, let's try all x in range, but order them heuristically
        # Solutions tend to exist for many x values, so just iterate
        
        for x in range(x_start, x_end + 1):
            # Time check in inner loop
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
            
            # Get factorization of B
            b_factors = get_factorization(B)
            b_sq_factors = [(p, 2 * e) for p, e in b_factors]
            
            # Get divisors of B^2
            divisors = get_divisors_from_factorization(b_sq_factors)
            
            # Sort divisors to allow early break
            divisors.sort()
            
            # For each divisor u of B^2 with u <= B:
            # v = B^2 // u
            # y = (u + B) / A must be integer
            # z = (v + B) / A must be integer
            # Check x < y < z
            
            for u in divisors:
                if u > B:
                    break
                v = (B * B) // u
                
                num_y = u + B
                if num_y % A != 0:
                    continue
                y = num_y // A
                
                num_z = v + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                
                # Check distinctness and ordering: x < y < z
                if x < y < z:
                    # Verify the equation holds (sanity check)
                    # 1/x + 1/y + 1/z should equal 4/n
                    # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
                    lhs = n * (y * z + x * z + x * y)
                    rhs = 4 * x * y * z
                    if lhs == rhs:
                        found = [x, y, z]
                        break
            
            if found:
                break
        
        answers[idx] = found
        
    return answers
