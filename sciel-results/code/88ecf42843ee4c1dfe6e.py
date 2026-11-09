import time
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    target = 479001600  # 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    target_exps = (10, 5, 2, 1, 1)
    
    # Generate all divisors of 12! that are > 12
    divisors = []
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        d = (2 ** e2) * (3 ** e3) * (5 ** e5) * (7 ** e7) * (11 ** e11)
                        if d > 12:
                            divisors.append(d)
    
    # Remove duplicates and sort
    divisors = sorted(set(divisors))
    n_divs = len(divisors)
    
    # Map value to exponents
    val_to_exps = {}
    for d in divisors:
        temp = d
        e2 = 0
        while temp % 2 == 0:
            e2 += 1
            temp //= 2
        e3 = 0
        while temp % 3 == 0:
            e3 += 1
            temp //= 3
        e5 = 0
        while temp % 5 == 0:
            e5 += 1
            temp //= 5
        e7 = 0
        while temp % 7 == 0:
            e7 += 1
            temp //= 7
        e11 = 0
        while temp % 11 == 0:
            e11 += 1
            temp //= 11
        val_to_exps[d] = (e2, e3, e5, e7, e11)
    
    # Precompute which divisors can be used given remaining exponents
    # For each divisor, store its exponents tuple
    div_exps = [(val_to_exps[d], d) for d in divisors]
    
    # DFS with aggressive pruning and time checking
    best_solution = [target]
    best_max = target
    
    def can_form(rem_exps, max_bound, used_set):
        """Check if we can form the remaining product with distinct factors <= max_bound."""
        if rem_exps == (0, 0, 0, 0, 0):
            return True
        if time.time() > deadline:
            return False
        
        rem_prod = (2 ** rem_exps[0]) * (3 ** rem_exps[1]) * (5 ** rem_exps[2]) * (7 ** rem_exps[3]) * (11 ** rem_exps[4])
        
        # If remaining product is 1, we're done
        if rem_prod == 1:
            return True
        
        # If remaining product < 13, we can't form it with a factor > 12
        if rem_prod < 13:
            return False
        
        # Try each divisor in ascending order
        for d_exps, d in div_exps:
            if d > max_bound:
                break
            if d in used_set:
                continue
            # Check if d divides remaining
            if (d_exps[0] <= rem_exps[0] and d_exps[1] <= rem_exps[1] and 
                d_exps[2] <= rem_exps[2] and d_exps[3] <= rem_exps[3] and 
                d_exps[4] <= rem_exps[4]):
                new_rem = (rem_exps[0] - d_exps[0], rem_exps[1] - d_exps[1],
                           rem_exps[2] - d_exps[2], rem_exps[3] - d_exps[3],
                           rem_exps[4] - d_exps[4])
                used_set.add(d)
                if can_form(new_rem, max_bound, used_set):
                    used_set.remove(d)
                    return True
                used_set.remove(d)
        return False
    
    def find_solution(rem_exps, max_bound, used_set, current_factors):
        """Find a solution with all factors <= max_bound."""
        if rem_exps == (0, 0, 0, 0, 0):
            return list(current_factors)
        if time.time() > deadline:
            return None
        if len(current_factors) >= 12:
            return None
        
        rem_prod = (2 ** rem_exps[0]) * (3 ** rem_exps[1]) * (5 ** rem_exps[2]) * (7 ** rem_exps[3]) * (11 ** rem_exps[4])
        
        if rem_prod == 1:
            return list(current_factors)
        if rem_prod < 13:
            return None
        
        for d_exps, d in div_exps:
            if d > max_bound:
                break
            if d in used_set:
                continue
            if (d_exps[0] <= rem_exps[0] and d_exps[1] <= rem_exps[1] and 
                d_exps[2] <= rem_exps[2] and d_exps[3] <= rem_exps[3] and 
                d_exps[4] <= rem_exps[4]):
                new_rem = (rem_exps[0] - d_exps[0], rem_exps[1] - d_exps[1],
                           rem_exps[2] - d_exps[2], rem_exps[3] - d_exps[3],
                           rem_exps[4] - d_exps[4])
                used_set.add(d)
                result = find_solution(new_rem, max_bound, used_set, current_factors + [d])
                used_set.remove(d)
                if result is not None:
                    return result
        return None
    
    # Binary search on the maximum factor
    # Lower bound: at least 13 (smallest factor > 12)
    # Upper bound: target (single factor)
    
    # First, try to find a solution with a reasonable upper bound
    # Let's try bounds in increasing order
    test_bounds = []
    # Generate candidate bounds: all divisors > 12, sorted
    for d in divisors:
        test_bounds.append(d)
    
    # Binary search
    lo = 0
    hi = len(test_bounds) - 1
    best_bound = target
    
    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        bound = test_bounds[mid]
        
        # Try to find a solution with max factor <= bound
        used_set = set()
        solution = find_solution(target_exps, bound, used_set, [])
        
        if solution is not None:
            best_bound = bound
            best_solution = solution
            hi = mid - 1
        else:
            lo = mid + 1
    
    # If binary search didn't find anything, try a direct DFS with the best bound found
    if best_solution == [target]:
        # Try a few specific bounds
        for bound in [200, 300, 500, 1000, 2000, 5000, 10000]:
            if time.time() > deadline:
                break
            used_set = set()
            solution = find_solution(target_exps, bound, used_set, [])
            if solution is not None:
                best_solution = solution
                best_max = max(solution)
                break
    
    return best_solution
