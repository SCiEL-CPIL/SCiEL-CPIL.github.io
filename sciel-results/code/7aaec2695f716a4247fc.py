import math
import time

# Precompute 12!
T12 = math.factorial(12)

# Prime factorization of 12!: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
primes = [2, 3, 5, 7, 11]
target_exponents = [10, 5, 2, 1, 1]

def generate_divisors_with_exponents():
    """
    Generate all divisors of 12! greater than 12, along with their prime factorization exponents.
    Returns: list of (divisor, [exp_2, exp_3, exp_5, exp_7, exp_11])
    """
    divisors = [(1, [0, 0, 0, 0, 0])]
    for idx, (p, e) in enumerate(zip(primes, target_exponents)):
        new_divs = []
        p_power = 1
        for exp in range(e + 1):
            for d, exps in divisors:
                new_exps = exps.copy()
                new_exps[idx] = exp
                new_divs.append((d * p_power, new_exps))
            p_power *= p
        divisors = new_divs
    
    # Filter: > 12, and sort by divisor
    result = sorted([(d, exps) for d, exps in divisors if d > 12], key=lambda x: x[0])
    return result

# Precompute all valid divisors with their prime factorizations
all_divisors_with_exps = generate_divisors_with_exponents()
all_divisors = [d for d, _ in all_divisors_with_exps]
divisor_to_exps = {d: exps for d, exps in all_divisors_with_exps}

def can_decompose(remaining, max_factor, used, min_index, depth, max_depth, deadline):
    """
    Backtracking search: can we decompose 'remaining' into distinct divisors of T12,
    all > 12, all <= max_factor, all with index >= min_index (to ensure distinctness and ordering),
    using at most (max_depth - depth) more factors?
    
    Returns a list of factors if found, None otherwise.
    """
    if time.time() > deadline:
        return None
    
    if remaining == 1:
        return list(used)
    
    if depth >= max_depth:
        return None
    
    # Pruning: if remaining > max_factor^(max_depth - depth), we can't reach it
    # since each remaining factor is at most max_factor
    if max_factor ** (max_depth - depth) < remaining:
        return None
    
    # Pruning: if remaining is not divisible by any valid divisor, fail
    # Also, remaining must be a divisor of T12 (it always is if we start from T12 and divide by divisors)
    
    for i in range(min_index, len(all_divisors)):
        d = all_divisors[i]
        if d > max_factor:
            break
        if d > remaining:
            break
        if remaining % d != 0:
            continue
        
        new_remaining = remaining // d
        # Pruning: new_remaining must be >= 1 and if it's > 1, it must be decomposable
        # with factors > 12 and <= max_factor, and distinct from what we've used
        
        result = can_decompose(new_remaining, max_factor, used + [d], i + 1, depth + 1, max_depth, deadline)
        if result is not None:
            return result
    
    return None

def find_decomposition(max_factor, deadline):
    """
    Try to find a decomposition of T12 into distinct factors all in (12, max_factor].
    Uses backtracking with increasing depth (number of factors).
    """
    # Try different numbers of factors, from 2 up to 12
    # For each, try to find a valid decomposition
    for k in range(2, 13):
        if time.time() > deadline:
            break
        result = can_decompose(T12, max_factor, [], 0, 0, k, deadline)
        if result is not None:
            return result
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin before 90s kill
    
    # First, find a good initial solution with 2 factors
    best_solution = [T12]
    best_max = T12
    
    # Find best 2-factor split
    sqrt_t12 = int(math.isqrt(T12))
    for d in range(sqrt_t12, 12, -1):
        if T12 % d == 0:
            other = T12 // d
            if d > 12 and other > 12 and d != other:
                best_solution = sorted([d, other])
                best_max = max(d, other)
                break
    
    # Now binary search on the maximum factor
    lo = 13
    hi = best_max
    
    while lo <= hi and (time.time() - start_time) < 80:
        mid = (lo + hi) // 2
        
        result = find_decomposition(mid, deadline)
        
        if result is not None:
            best_solution = sorted(result)
            best_max = max(result)
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Verify the best solution
    if best_solution:
        prod = 1
        for d in best_solution:
            prod *= d
        if prod == T12 and 1 <= len(best_solution) <= 12 and all(d > 12 for d in best_solution):
            if len(set(best_solution)) == len(best_solution):
                return best_solution
    
    # Fallback
    return [T12]

if __name__ == "__main__":
    res = construct()
    print(res)
    print(f"Max factor: {max(res) if res else 'N/A'}")
    print(f"Product: {math.prod(res) if res else 'N/A'}")
