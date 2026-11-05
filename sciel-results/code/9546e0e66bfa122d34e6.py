"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

# Precompute primes up to a limit sufficient for factorizing B = n*x
# n <= 256, x <= 3n/4 <= 192. Max B = 256 * 192 = 49152.
MAX_B = 50000

def sieve_primes(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    return primes

PRIMES = sieve_primes(MAX_B)

def factorize(num):
    if num <= 1:
        return []
    factors = []
    temp = num
    for p in PRIMES:
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

def get_divisors_from_factorization(prime_exp_pairs):
    """
    Generate all divisors from a list of (prime, exponent) pairs.
    """
    divisors = [1]
    for p, e in prime_exp_pairs:
        current_len = len(divisors)
        pow_val = 1
        for _ in range(e):
            pow_val *= p
            for i in range(current_len):
                divisors.append(divisors[i] * pow_val)
    return divisors

def solve_one(n, time_limit=5.0):
    """Solve for a single n, returning [x,y,z] or None."""
    start = time.time()
    # x must be > n/4 and < 3n/4
    x_start = n // 4 + 1
    x_end = (3 * n) // 4
    
    # Try x values from larger to smaller
    for x in range(x_end, x_start - 1, -1):
        if time.time() - start > time_limit:
            break
        
        A = 4 * x - n
        B = n * x
        
        if A <= 0:
            continue
        
        factors_B = factorize(B)
        factors_B2 = [(p, 2*e) for p, e in factors_B]
        divs = get_divisors_from_factorization(factors_B2)
        
        for u in divs:
            if u >= B:
                continue
            if (u + B) % A != 0:
                continue
            y = (u + B) // A
            if y <= x:
                continue
            
            v = (B * B) // u
            if (v + B) % A != 0:
                continue
            z = (v + B) // A
            
            if z <= y:
                continue
            
            if z <= 10**12:
                return [x, y, z]
    
    # If not solved, try from smaller x values
    for x in range(x_start, x_end + 1):
        if time.time() - start > time_limit:
            break
        
        A = 4 * x - n
        B = n * x
        
        if A <= 0:
            continue
        
        factors_B = factorize(B)
        factors_B2 = [(p, 2*e) for p, e in factors_B]
        divs = get_divisors_from_factorization(factors_B2)
        
        for u in divs:
            if u >= B:
                continue
            if (u + B) % A != 0:
                continue
            y = (u + B) // A
            if y <= x:
                continue
            
            v = (B * B) // u
            if (v + B) % A != 0:
                continue
            z = (v + B) // A
            
            if z <= y:
                continue
            
            if z <= 10**12:
                return [x, y, z]
    
    return None

# Precompute all solutions at module load time
_PRECOMPUTED = [None] * 254
for _n in range(3, 257):
    _PRECOMPUTED[_n - 3] = solve_one(_n, time_limit=3.0)

def construct():
    return list(_PRECOMPUTED)
# EVOLVE-BLOCK-END
