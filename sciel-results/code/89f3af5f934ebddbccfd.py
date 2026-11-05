"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    N = math.factorial(12)
    
    # Generate divisors of N
    divisors = set()
    for i in range(1, int(math.isqrt(N)) + 1):
        if N % i == 0:
            divisors.add(i)
            divisors.add(N // i)
            
    # Filter > 12 and sort
    valid_divs = sorted([d for d in divisors if d > 12])
    
    best_max = N
    best_factors = [N]
    
    def dfs(rem, last_factor, current_max, path):
        nonlocal best_max, best_factors
        
        if current_max >= best_max:
            return
            
        if rem == 1:
            if current_max < best_max:
                best_max = current_max
                best_factors = path[:]
            return
        
        # Option 1: rem is the last factor
        if rem > last_factor and rem > 12:
            new_max = max(current_max, rem)
            path.append(rem)
            dfs(1, rem, new_max, path)
            path.pop()
            
        # Option 2: split rem into x and rem//x
        limit = int(math.isqrt(rem))
        
        for x in valid_divs:
            if x <= last_factor:
                continue
            if x > limit:
                break
            
            if rem % x == 0:
                new_rem = rem // x
                new_max = max(current_max, x)
                path.append(x)
                dfs(new_rem, x, new_max, path)
                path.pop()
                
    dfs(N, 12, 0, [])
    
    return best_factors
# EVOLVE-BLOCK-END
