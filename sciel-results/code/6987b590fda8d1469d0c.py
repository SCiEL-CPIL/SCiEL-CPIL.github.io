import time
import math
from functools import lru_cache
from fractions import Fraction

def construct():
    answers = [None] * 254
    LIMIT = 10**12
    start_time = time.time()
    DEADLINE = 82.0  # Safety margin before 90s limit

    # Precompute small prime list for fast factorization
    SMALL_PRIMES = []
    def _sieve(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i, v in enumerate(sieve) if v]
    SMALL_PRIMES = _sieve(10000)

    def factorize(n):
        """Factorize n, return dict of prime->exponent."""
        factors = {}
        temp = n
        for p in SMALL_PRIMES:
            if p * p > temp:
                break
            while temp % p == 0:
                factors[p] = factors.get(p, 0) + 1
                temp //= p
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        return factors

    @lru_cache(maxsize=None)
    def get_divisors_of_square(n):
        """
        Returns tuple of divisors of n^2.
        n is small enough to factor quickly.
        """
        if n == 1:
            return (1,)
        
        factors = factorize(n)
        
        # Generate divisors of n^2
        # If n = prod p_i^e_i, n^2 = prod p_i^(2e_i)
        # Divisors are prod p_i^(k_i) where 0 <= k_i <= 2e_i
        divs = [1]
        for p, e in factors.items():
            max_exp = 2 * e
            current_divs = []
            for d in divs:
                p_power = 1
                for _ in range(max_exp + 1):
                    current_divs.append(d * p_power)
                    p_power *= p
            divs = current_divs
        return tuple(divs)

    def find_solution_for_n(n):
        """Find a valid (x,y,z) with x<y<z, 1/x+1/y+1/z=4/n, all <= LIMIT."""
        # x range: n/4 < x < 3n/4
        start_x = n // 4 + 1
        end_x = (3 * n) // 4 - 1
        
        if start_x > end_x:
            return None
        
        # Generate candidate x values
        x_candidates = list(range(start_x, end_x + 1))
        
        # Sort by A = 4x - n (smaller A first)
        def x_priority(x):
            A = 4 * x - n
            return A
        
        x_candidates.sort(key=x_priority)
        
        for x in x_candidates:
            if time.time() - start_time > DEADLINE:
                return None
                
            A = 4 * x - n
            B = n * x
            if A <= 0:
                continue
            
            # We need U such that U * V = B^2, U < V
            # U = A*y - B
            # y = (U+B)/A
            # y > x => U > Ax - B
            # z = (V+B)/A
            # z > y => V > U
            # z <= LIMIT => V <= A*LIMIT - B
            
            # Get divisors of B^2 (cached)
            divs = get_divisors_of_square(B)
            
            # Filter divisors U
            # U must be < B (since U < V and UV=B^2)
            # Also U > Ax - B (for y > x)
            lower_bound_U = A * x - B
            upper_bound_U = B - 1  # U < B
            
            if lower_bound_U >= upper_bound_U:
                continue
            
            # Filter divisors in valid range
            valid_divs = [U for U in divs if lower_bound_U < U <= upper_bound_U]
            if not valid_divs:
                continue
            
            for U in valid_divs:
                # Check if y is integer
                if (U + B) % A != 0:
                    continue
                y = (U + B) // A
                if y <= x:
                    continue
                
                V = (B * B) // U
                # Check if z is integer
                if (V + B) % A != 0:
                    continue
                z = (V + B) // A
                if z <= y:
                    continue
                if z > LIMIT:
                    continue
                
                return [x, y, z]
        
        return None

    for idx, n in enumerate(range(3, 257)):
        if time.time() - start_time > DEADLINE:
            break
        
        answers[idx] = find_solution_for_n(n)
    
    return answers
