import math
import time
import numpy as np
from functools import lru_cache

# Precompute 12!
T12 = math.factorial(12)

# Prime factorization of 12!: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
primes = [2, 3, 5, 7, 11]
target_exponents = [10, 5, 2, 1, 1]

def generate_divisors():
    """Generate all divisors of 12! greater than 12, sorted."""
    divisors = [1]
    for p, e in zip(primes, target_exponents):
        new_divs = []
        p_power = 1
        for _ in range(e + 1):
            for d in divisors:
                new_divs.append(d * p_power)
            p_power *= p
        divisors = new_divs
    
    # Filter: > 12, and sort
    result = sorted([d for d in divisors if d > 12])
    return result

# Precompute all valid divisors
all_divisors = generate_divisors()
divisor_set = set(all_divisors)

def can_decompose(target, max_factor, min_factor=13, max_factors=12, used=None):
    """
    Check if we can decompose `target` into distinct factors in [min_factor, max_factor].
    Returns a list of factors if found, None otherwise.
    Uses DFS with pruning.
    """
    if used is None:
        used = set()
    
    if target == 1:
        return []
    
    if target <= max_factor and target >= min_factor and target not in used:
        return [target]
    
    # Try factors from max_factor down to min_factor (larger first for pruning)
    # But we need to be smart: only try divisors of target
    # Get divisors of target that are in our allowed range
    best = None
    # We need to find divisors of target in [min_factor, max_factor] not in used
    # Iterate through potential factors
    # To prune: if we pick factor f, remaining = target/f must be decomposable
    # with factors > f (to ensure distinctness and ordering)
    
    # Get all divisors of target
    divs_of_target = []
    for i in range(min_factor, min(max_factor, target) + 1):
        if target % i == 0 and i not in used:
            divs_of_target.append(i)
    
    # Sort in descending order to try larger factors first (better pruning)
    divs_of_target.sort(reverse=True)
    
    for f in divs_of_target:
        if f < min_factor:
            break
        remaining = target // f
        if remaining == 1:
            # We've used all factors
            return [f]
        # Check if remaining can be formed by distinct factors all > f and <= max_factor
        # Actually, we need factors > 12 and <= max_factor, distinct from used
        # For ordering, let's require next factors > f to avoid permutations
        # But that might miss solutions... Let's just use the used set
        result = can_decompose(remaining, max_factor, min_factor, max_factors, used | {f})
        if result is not None:
            return [f] + result
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin before 90s kill
    
    # Binary search on the maximum factor T
    # Lower bound: at least 13
    # Upper bound: T12 (trivial solution)
    
    lo = 13
    hi = T12
    best_solution = [T12]  # Fallback: single factor
    
    # First, try to find a good solution with a heuristic
    # Try small number of factors first
    
    # Heuristic: try 2 factors first
    # Find divisor d of T12 such that d > 12 and T12//d > 12 and d != T12//d
    # Minimize max(d, T12//d)
    best_2factor = None
    sqrt_t12 = int(math.isqrt(T12))
    for d in range(sqrt_t12, 12, -1):
        if T12 % d == 0:
            other = T12 // d
            if d > 12 and other > 12 and d != other:
                if best_2factor is None or max(d, other) < max(best_2factor[0], best_2factor[1]):
                    best_2factor = (d, other)
                break  # First one found going down from sqrt is optimal for 2 factors
    
    if best_2factor:
        best_solution = sorted(best_2factor)
        hi = max(best_2factor)
    
    # Try 3 factors
    # This is more complex, let's use the DFS approach with binary search on T
    
    # Binary search on T
    while lo <= hi and (time.time() - start_time) < 80:
        mid = (lo + hi) // 2
        
        # Check if we can decompose T12 into distinct factors all in (12, mid]
        result = can_decompose(T12, mid, 13, 12, set())
        
        if result is not None:
            best_solution = sorted(result)
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Verify the best solution
    if best_solution:
        prod = 1
        for d in best_solution:
            prod *= d
        if prod == T12 and len(best_solution) <= 12 and all(d > 12 for d in best_solution):
            # Check distinctness
            if len(set(best_solution)) == len(best_solution):
                return best_solution
    
    # Fallback
    return [T12]

if __name__ == "__main__":
    res = construct()
    print(res)
    print(f"Max factor: {max(res) if res else 'N/A'}")
    print(f"Product: {math.prod(res) if res else 'N/A'}")
