"""Rewritten program: binary search on the maximum factor M with feasibility check."""
# EVOLVE-BLOCK-START
import math
import time

T12 = math.factorial(12)  # 479001600

# Prime factorization of 12!
# 12! = 2^10 * 3^5 * 5^2 * 7 * 11
PRIMES = [2, 3, 5, 7, 11]
INITIAL_EXP = (10, 5, 2, 1, 1)

def exp_to_val(exp):
    """Convert exponent tuple to integer value."""
    result = 1
    for i, e in enumerate(exp):
        if e > 0:
            result *= PRIMES[i] ** e
    return result

def val_to_exp(n):
    """Convert integer to exponent tuple for the known primes. Returns None if n has other prime factors."""
    exp = [0, 0, 0, 0, 0]
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            exp[i] += 1
            n //= p
    if n != 1:
        return None
    return tuple(exp)

def solve():
    """
    Find distinct factors > 12 multiplying to 12! minimizing the largest factor.
    Uses binary search on the maximum factor M, with a feasibility check for each M.
    """
    deadline = time.time() + 55
    
    # Precompute divisors for all possible exponent states
    divisor_cache = {}
    
    def generate_divisors(exp):
        """Generate all divisors (as values) of the number represented by exp."""
        divs = [1]
        for i, e in enumerate(exp):
            if e > 0:
                p = PRIMES[i]
                powers = [1]
                for k in range(1, e + 1):
                    powers.append(powers[-1] * p)
                new_divs = []
                for d in divs:
                    for pow_val in powers:
                        new_divs.append(d * pow_val)
                divs = new_divs
        return sorted(divs)

    # Precompute all divisors for all possible states
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        exp = (e2, e3, e5, e7, e11)
                        divisor_cache[exp] = generate_divisors(exp)

    # Memoization for feasibility check
    memo_feasible = {}
    memo_solution = {}
    
    def feasible(rem_exp, last_f, M):
        """
        Check if we can decompose rem_exp into distinct factors all > last_f, all <= M.
        Returns True if feasible, False otherwise.
        """
        if time.time() > deadline:
            return False
            
        if rem_exp == (0, 0, 0, 0, 0):
            return True
            
        # Key for memoization
        key = (rem_exp, last_f, M)
        if key in memo_feasible:
            return memo_feasible[key]
            
        rem_val = exp_to_val(rem_exp)
        
        # If rem_val is a valid single factor
        if rem_val > last_f and rem_val >= 13 and rem_val <= M:
            memo_feasible[key] = True
            return True
            
        # Try to find a factor d of rem_val such that:
        # 1. d > last_f
        # 2. d >= 13
        # 3. d <= M
        # 4. The remaining part R = rem_val // d can be decomposed into distinct factors > d, all <= M.
        
        divs = divisor_cache[rem_exp]
        
        for d in divs:
            if d < 13:
                continue
            if d <= last_f:
                continue
            if d > M:
                break
            if d == rem_val:
                continue
            
            R = rem_val // d
            
            # Pruning: next factors must be > d, so minimum next factor is d+1
            if R < d + 1:
                continue
            
            R_exp = val_to_exp(R)
            if R_exp is None:
                continue
                
            if feasible(R_exp, d, M):
                memo_feasible[key] = True
                return True
        
        memo_feasible[key] = False
        return False

    def find_solution(rem_exp, last_f, M):
        """
        Find a decomposition of rem_exp into distinct factors > last_f, all <= M.
        Returns a list of factors or None.
        """
        if time.time() > deadline:
            return None
            
        if rem_exp == (0, 0, 0, 0, 0):
            return []
            
        key = (rem_exp, last_f, M)
        if key in memo_solution:
            return memo_solution[key]
            
        rem_val = exp_to_val(rem_exp)
        
        # If rem_val is a valid single factor
        if rem_val > last_f and rem_val >= 13 and rem_val <= M:
            memo_solution[key] = [rem_val]
            return [rem_val]
            
        divs = divisor_cache[rem_exp]
        
        for d in divs:
            if d < 13:
                continue
            if d <= last_f:
                continue
            if d > M:
                break
            if d == rem_val:
                continue
            
            R = rem_val // d
            
            # Pruning: next factors must be > d
            if R < d + 1:
                continue
            
            R_exp = val_to_exp(R)
            if R_exp is None:
                continue
            
            # Check feasibility before recursing to avoid wasted work
            if not feasible(R_exp, d, M):
                continue
            
            sub_sol = find_solution(R_exp, d, M)
            if sub_sol is not None:
                result = [d] + sub_sol
                memo_solution[key] = result
                return result
        
        memo_solution[key] = None
        return None

    # Compute lower bound for M
    prod = 1
    max_k = 0
    for k in range(1, 13):
        next_factor = 12 + k
        if prod * next_factor > T12:
            break
        prod *= next_factor
        max_k = k
    
    lower_bound = 12 + max_k
    
    # Binary search for the smallest M
    lo = lower_bound
    hi = T12
    best_solution = [T12]
    
    # First, check if hi is feasible (it should be)
    # Then binary search
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        
        if feasible(INITIAL_EXP, 0, mid):
            best_solution = find_solution(INITIAL_EXP, 0, mid)
            if best_solution is not None:
                hi = mid - 1
            else:
                # Feasible but couldn't find solution due to time, try larger M
                lo = mid + 1
        else:
            lo = mid + 1
    
    # If we found a solution, return it
    if best_solution is not None and best_solution != [T12]:
        product = 1
        for f in best_solution:
            product *= f
        if product == T12 and all(f > 12 for f in best_solution) and len(set(best_solution)) == len(best_solution) and 1 <= len(best_solution) <= 12:
            return best_solution
    
    # Try linear search from lower_bound if binary search didn't work well
    for M in range(lower_bound, min(lower_bound + 5000, T12 + 1)):
        if time.time() > deadline:
            break
        sol = find_solution(INITIAL_EXP, 0, M)
        if sol is not None:
            product = 1
            for f in sol:
                product *= f
            if product == T12 and all(f > 12 for f in sol) and len(set(sol)) == len(sol) and 1 <= len(sol) <= 12:
                return sol
    
    # Fallback: return the trivial solution
    return [T12]

def construct():
    """
    Return 1..12 DISTINCT integers all greater than 12, with product exactly 12!.
    Minimize the LARGEST factor.
    """
    try:
        solution = solve()
        # Verify
        product = 1
        for f in solution:
            product *= f
        if product == T12 and all(f > 12 for f in solution) and len(set(solution)) == len(solution) and 1 <= len(solution) <= 12:
            return solution
    except Exception:
        pass
    
    # Fallback
    return [T12]

# EVOLVE-BLOCK-END
