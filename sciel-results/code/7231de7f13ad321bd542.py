import time
import math
from functools import lru_cache

def construct():
    answers = [None] * 254
    LIMIT = 10**12
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s limit
    
    @lru_cache(maxsize=None)
    def get_divisors_of_square(n):
        """
        Returns tuple of divisors of n^2.
        n is small enough to factor quickly.
        """
        if n == 1:
            return (1,)
        
        # Factorize n
        factors = {}
        temp = n
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                factors[d] = factors.get(d, 0) + 1
                temp //= d
            d += 1
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
            
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
        
        # Generate candidate x values, sorted by heuristic:
        # Prefer x where A=4x-n is small (easier divisibility) and B=nx has many divisors
        x_candidates = list(range(start_x, end_x + 1))
        
        # Sort by A = 4x - n (smaller A first), then by number of prime factors of B
        def x_priority(x):
            A = 4 * x - n
            B = n * x
            # Count distinct prime factors of B as a proxy for divisor count
            temp = B
            count = 0
            d = 2
            while d * d <= temp:
                while temp % d == 0:
                    count += 1
                    temp //= d
                d += 1
            if temp > 1:
                count += 1
            return (A, -count)
        
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
