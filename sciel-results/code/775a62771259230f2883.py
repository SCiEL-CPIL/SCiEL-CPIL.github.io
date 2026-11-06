import time
from math import factorial

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    N = factorial(12)  # 479001600
    
    # Generate all divisors of N that are > 12
    divisors = set()
    i = 1
    while i * i <= N:
        if N % i == 0:
            if i > 12:
                divisors.add(i)
            j = N // i
            if j > 12:
                divisors.add(j)
        i += 1
    unique_divs = sorted(divisors)
    
    best_max = N  # initial: single factor [N]
    best_solution = [N]
    
    def backtrack(remaining, start_idx, current_factors, current_max):
        nonlocal best_max, best_solution
        if time.time() - t0 > DEADLINE:
            return
        if remaining == 1:
            if current_max < best_max:
                best_max = current_max
                best_solution = list(current_factors)
            return
        
        # If we use remaining as a single factor, that's a valid solution
        # (remaining > 12 is guaranteed since all proper divisors > 12 are used,
        #  and remaining starts at N and we only divide by factors > 12)
        # Actually remaining could be <= 12 if we've divided too much, but we only
        # divide by factors > 12, so remaining < N. Let's check: if remaining <= 12,
        # we can't use it as a factor, so this branch is dead.
        if remaining <= 12:
            return
        
        # Option: use remaining as the last factor
        new_max = max(current_max, remaining)
        if new_max < best_max:
            current_factors.append(remaining)
            backtrack(1, len(unique_divs), current_factors, new_max)
            current_factors.pop()
        
        # Try splitting remaining into d * (remaining/d) where d > 12, remaining/d > 12
        for i in range(start_idx, len(unique_divs)):
            d = unique_divs[i]
            if d > remaining:
                break
            if remaining % d != 0:
                continue
            new_remaining = remaining // d
            if new_remaining <= 12:
                break  # since d is increasing, new_remaining will only get smaller
            new_max = max(current_max, d)
            if new_max >= best_max:
                break  # larger d will only increase or maintain new_max
            
            current_factors.append(d)
            backtrack(new_remaining, i + 1, current_factors, new_max)
            current_factors.pop()
    
    backtrack(N, 0, [], 0)
    
    return best_solution
