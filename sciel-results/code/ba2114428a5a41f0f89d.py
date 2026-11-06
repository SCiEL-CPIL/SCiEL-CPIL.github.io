import time
import math

# Precompute primes up to 500 for trial division of B (max B ~ 50000)
def _get_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

_PRIMES = _get_primes(500)

def _factorize(n):
    """Return list of (prime, exponent) for n."""
    factors = []
    temp_n = n
    for p in _PRIMES:
        if p * p > temp_n:
            break
        if temp_n % p == 0:
            exp = 0
            while temp_n % p == 0:
                temp_n //= p
                exp += 1
            factors.append((p, exp))
    if temp_n > 1:
        factors.append((temp_n, 1))
    return factors

def _get_divisors_of_square(factors):
    """
    Given factorization of B as [(p, e)], generate divisors of B^2.
    B^2 has factors [(p, 2e)].
    """
    divisors = [1]
    for p, e in factors:
        # Exponent in B^2 is 2*e
        max_exp = 2 * e
        new_divs = []
        for d in divisors:
            pe = 1
            for _ in range(max_exp + 1):
                new_divs.append(d * pe)
                pe *= p
        divisors = new_divs
    return divisors

def construct():
    deadline = time.time() + 85
    results = [None] * 254
    BOUND = 10**12
    
    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        
        found = None
        # x must be in (n/4, 3n/4)
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if found:
                break
            if time.time() > deadline:
                break
            
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # We need (A*y - B)(A*z - B) = B^2
            # Let d = A*y - B. Then d is a divisor of B^2.
            # y = (d + B) / A
            # z = (B^2/d + B) / A
            # Conditions:
            # 1. d < B (to ensure y < z, since A*y-B < A*z-B)
            # 2. (d + B) % A == 0
            # 3. y > x
            # 4. z > y
            # 5. z <= BOUND
            
            # Factor B to get divisors of B^2
            fac_B = _factorize(B)
            divs = _get_divisors_of_square(fac_B)
            
            # Optimization: We only care about d < B
            # Sorting divisors is expensive if there are many, but count is small.
            # We can just iterate and filter.
            
            for d in divs:
                if d >= B:
                    continue
                
                num_y = d + B
                if num_y % A != 0:
                    continue
                
                y = num_y // A
                if y <= x:
                    continue
                
                # Calculate z
                # A*z - B = B^2 / d
                # A*z = B^2 / d + B
                # z = (B^2 / d + B) / A
                
                # Check if B^2 is divisible by d (it is, by construction)
                term1 = (B * B) // d
                num_z = term1 + B
                
                if num_z % A != 0:
                    continue
                
                z = num_z // A
                
                if z <= y:
                    continue
                if z > BOUND:
                    continue
                
                # Final verification
                # 1/x + 1/y + 1/z = 4/n
                # n(yz + xz + xy) = 4xyz
                if n * (y * z + x * z + x * y) == 4 * x * y * z:
                    found = [x, y, z]
                    break
        
        results[idx] = found
    
    return results
