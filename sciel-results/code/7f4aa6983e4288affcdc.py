"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial, log
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    target = factorial(12)  # 479001600
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    
    best = [target]  # fallback: single factor
    best_max = target
    
    # Precompute all divisors of target that are > 12
    def get_all_divisors(n):
        divs = set()
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return sorted(divs)
    
    all_divisors = get_all_divisors(target)
    valid_divisors = [d for d in all_divisors if d > 12]
    
    # Strategy: DFS building factors in ascending order to naturally minimize max
    # We enforce that each new factor >= previous factor (non-decreasing order)
    # This avoids permutations and helps find balanced solutions
    
    def search(remaining, min_factor, used_set, current_factors, max_so_far):
        nonlocal best, best_max, start_time
        
        if time.time() - start_time > DEADLINE:
            return
        
        if len(current_factors) >= 12:
            if remaining == 1 and len(current_factors) >= 1:
                mx = max(current_factors)
                if mx < best_max:
                    best_max = mx
                    best = list(current_factors)
            return
        
        if remaining == 1:
            if len(current_factors) >= 1:
                mx = max(current_factors)
                if mx < best_max:
                    best_max = mx
                    best = list(current_factors)
            return
        
        # Pruning: if max_so_far >= best_max, no improvement possible
        if max_so_far >= best_max:
            return
        
        # If we can't fit remaining as a single factor > 12, and we need more factors,
        # check if remaining itself could work as the last factor
        # Actually, let's just find valid divisors of remaining >= min_factor
        
        # Find divisors of remaining that are >= min_factor, > 12, not used, and < best_max
        divs = []
        for d in valid_divisors:
            if d < min_factor:
                continue
            if d >= best_max:
                break  # sorted, so no more candidates
            if d in used_set:
                continue
            if remaining % d == 0:
                divs.append(d)
        
        # Sort to try smaller divisors first (ascending) for better balance
        # But also consider: trying divisors close to remaining^(1/k) might be better
        # Let's try ascending order first since it naturally keeps factors small
        
        # Limit search space
        max_divs_to_try = 150
        divs = divs[:max_divs_to_try]
        
        for d in divs:
            if d in used_set or d <= 12 or d < min_factor:
                continue
            new_max = max(max_so_far, d)
            if new_max >= best_max:
                continue
            
            new_used = used_set | {d}
            new_factors = current_factors + [d]
            search(remaining // d, d, new_used, new_factors, new_max)
            
            if time.time() - start_time > DEADLINE:
                return
    
    # Start search with min_factor = 13 (smallest valid factor)
    search(target, 13, set(), [], 0)
    
    return best
# EVOLVE-BLOCK-END
