"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

# Precompute 12!
T12 = math.factorial(12)  # 479001600

# Prime factorization of 12!
# 12! = 2^10 * 3^5 * 5^2 * 7 * 11
PRIMES = [2, 3, 5, 7, 11]
EXPONENTS = [10, 5, 2, 1, 1]


def get_all_divisors(n):
    """Generate all divisors of n using its prime factorization."""
    divisors = [1]
    for p, e in zip(PRIMES, EXPONENTS):
        new_divs = []
        for d in divisors:
            val = d
            for i in range(e + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    return sorted(divisors)


def precompute():
    """Precompute divisors > 12 and their exponent vectors."""
    all_divs = get_all_divisors(T12)
    # Filter divisors > 12
    valid_divs = [d for d in all_divs if d > 12]
    
    # For each valid divisor, compute its exponent vector
    div_exp = {}
    for d in valid_divs:
        exp = []
        temp = d
        for p in PRIMES:
            cnt = 0
            while temp % p == 0:
                cnt += 1
                temp //= p
            exp.append(cnt)
        div_exp[d] = exp
    
    return valid_divs, div_exp


def construct():
    """
    Return 1..12 DISTINCT integers all greater than 12, with product exactly 12!.
    Minimize the LARGEST factor.
    """
    start_time = time.time()
    deadline = start_time + 80  # 80s out of 90s
    
    # Precompute divisors
    valid_divs, div_exp = precompute()
    n_divs = len(valid_divs)
    
    # Target exponent vector
    target_exp = EXPONENTS[:]
    
    best_solution = [T12]
    best_max = T12
    
    # DFS over subsets of divisors, building in increasing order
    # State: (start_idx, remaining_exp, chosen_factors, current_max)
    # We iterate divisors in sorted order to ensure distinctness and canonical form
    
    def dfs_subset(start_idx, remaining_exp, chosen, current_max):
        nonlocal best_solution, best_max
        
        if time.time() > deadline:
            return
        
        # Check if remaining_exp is all zeros -> found a solution
        if all(e == 0 for e in remaining_exp):
            if chosen:
                if current_max < best_max:
                    best_max = current_max
                    best_solution = list(chosen)
            return
        
        # Pruning: if current_max >= best_max, no point continuing
        # (we can't improve)
        
        # Try adding each divisor from start_idx onward
        for i in range(start_idx, n_divs):
            if time.time() > deadline:
                return
            
            d = valid_divs[i]
            # If d >= best_max, we can't improve, so skip (and all subsequent are larger)
            if d >= best_max:
                break
            
            exp = div_exp[d]
            
            # Check if we can subtract this divisor's exponents from remaining
            can_subtract = True
            for j in range(len(PRIMES)):
                if remaining_exp[j] < exp[j]:
                    can_subtract = False
                    break
            
            if not can_subtract:
                continue
            
            # Subtract and recurse
            new_remaining = [remaining_exp[j] - exp[j] for j in range(len(PRIMES))]
            new_max = max(current_max, d)
            chosen.append(d)
            
            dfs_subset(i + 1, new_remaining, chosen, new_max)
            
            chosen.pop()
    
    # Start DFS
    dfs_subset(0, target_exp, [], 0)
    
    # Verify solution
    if best_solution:
        product = 1
        for f in best_solution:
            product *= f
        if product == T12 and all(f > 12 for f in best_solution) and len(set(best_solution)) == len(best_solution):
            return best_solution
    
    # Fallback: return the trivial solution
    return [T12]

# EVOLVE-BLOCK-END
