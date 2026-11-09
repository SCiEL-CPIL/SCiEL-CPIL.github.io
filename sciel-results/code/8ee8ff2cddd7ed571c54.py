import math
import time
import bisect

def construct():
    N = math.factorial(12)
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    # Get all divisors > 12
    divisors = []
    for i in range(1, int(math.isqrt(N)) + 1):
        if N % i == 0:
            if i > 12:
                divisors.append(i)
            j = N // i
            if j != i and j > 12:
                divisors.append(j)
    divisors.sort()
    
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def dfs(remaining, last_val, max_val, current_factors):
        # Time check
        if time.time() > deadline:
            return None
            
        if remaining == 1:
            return current_factors[:]
            
        if len(current_factors) == 12:
            return None
            
        # If remaining is prime, it must be the last factor
        if is_prime(remaining):
            if remaining > last_val and remaining <= max_val:
                current_factors.append(remaining)
                return current_factors
            else:
                return None

        # Find start index in divisors
        start_idx = bisect.bisect_right(divisors, last_val)
        
        for i in range(start_idx, len(divisors)):
            d = divisors[i]
            if d > max_val:
                break
            if d > remaining:
                break
            if remaining % d != 0:
                continue
                
            current_factors.append(d)
            result = dfs(remaining // d, d, max_val, current_factors)
            if result is not None:
                return result
            current_factors.pop()
            
        return None

    # Iterate M over divisors
    for M in divisors:
        if time.time() > deadline:
            break
        result = dfs(N, 12, M, [])
        if result is not None:
            return result
            
    return [N]
