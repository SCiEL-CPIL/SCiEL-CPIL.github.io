"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial

def construct():
    target = factorial(12)  # 479001600
    deadline = time.time() + 80
    
    # Generate all divisors of target
    def get_divisors(n):
        divs = {1}
        d = 2
        while d * d <= n:
            while n % d == 0:
                n //= d
                new_divs = set()
                for x in divs:
                    new_divs.add(x * d)
                divs |= new_divs
            d += 1
        if n > 1:
            new_divs = set()
            for x in divs:
                new_divs.add(x * n)
            divs |= new_divs
        return sorted(divs)
    
    all_divs = get_divisors(target)
    candidates = [d for d in all_divs if d > 12]
    
    best_max = target
    best_set = [target]
    
    # DFS with pruning
    def dfs(remaining, start_idx, current_factors, current_max):
        nonlocal best_max, best_set
        if time.time() > deadline:
            return
        if remaining == 1:
            if len(current_factors) >= 2:
                if current_max < best_max:
                    best_max = current_max
                    best_set = list(current_factors)
            return
        if len(current_factors) >= 12:
            return
        
        for i in range(start_idx, len(candidates)):
            f = candidates[i]
            if f > remaining:
                break
            if f >= best_max:
                break
            if remaining % f != 0:
                continue
            
            new_rem = remaining // f
            new_max = max(current_max, f)
            
            if new_rem == 1:
                if len(current_factors) + 1 >= 2:
                    if new_max < best_max:
                        best_max = new_max
                        best_set = list(current_factors) + [f]
            else:
                if i + 1 >= len(candidates):
                    continue
                next_min = candidates[i+1]
                if new_rem < next_min:
                    continue
                dfs(new_rem, i+1, current_factors + [f], new_max)
    
    dfs(target, 0, [], 12)
    
    if best_set:
        return best_set
    return [target]

# EVOLVE-BLOCK-END
