import time
from math import isqrt, gcd

def construct():
    start_time = time.time()
    deadline = start_time + 75.0
    
    MAX_N = 256
    
    # 1. Find all primes up to MAX_N
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(MAX_N)
    
    # 2. Prepare ILP data
    # Variables x_2, ..., x_256 (index 0 corresponds to n=2, etc.)
    ns = list(range(2, MAX_N + 1))
    num_vars = len(ns)
    
    # Objective coefficients: maximize sum(n * x_n)
    c = [-n for n in ns]  # scipy minimizes, so negate
    
    # Constraints: For each prime p, sum(x_n for n divisible by p) <= 1
    A_rows = []
    b_vals = []
    
    for p in primes:
        row = [0] * num_vars
        for i, n in enumerate(ns):
            if n % p == 0:
                row[i] = 1
        A_rows.append(row)
        b_vals.append(1)
        
    A = A_rows
    b = b_vals
    
    # Bounds: 0 <= x_i <= 1
    integrality = [1] * num_vars
    lb = [0] * num_vars
    ub = [1] * num_vars
    
    selected = []
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        constraint = LinearConstraint(A, lb=[0]*len(primes), ub=[1]*len(primes))
        bounds = Bounds(lb, ub)
        
        res = milp(c, constraints=constraint, integrality=integrality, bounds=bounds)
        
        if res.status == 0:  # Optimal
            selected = [ns[i] for i in range(num_vars) if res.x[i] > 0.5]
        else:
            # Fallback if ILP fails
            raise Exception("ILP failed")
            
    except Exception:
        # Fallback: Greedy approach
        remaining = set(range(2, MAX_N + 1))
        selected = []
        candidates = sorted(remaining, reverse=True)
        for n in candidates:
            if n not in remaining:
                continue
            is_coprime = True
            for s in selected:
                if gcd(n, s) != 1:
                    is_coprime = False
                    break
            if is_coprime:
                selected.append(n)
    
    # Local search refinement: try to improve by swapping or adding
    # Check if we can add 1 (it's coprime to everything)
    if 1 not in selected:
        selected.append(1)
    
    # Local search: try to improve by replacing one selected number with a better combination
    # Since we have at most 255 numbers, we can try pairwise swaps
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        selected_set = set(selected)
        
        # Try removing one element and seeing if we can add two better ones
        for i in range(len(selected)):
            if time.time() >= deadline - 5.0:
                break
            temp_selected = [s for j, s in enumerate(selected) if j != i]
            removed = selected[i]
            
            # Find all numbers not in selected that are coprime to all in temp_selected
            # and also coprime to each other (we need to pick at least 2)
            candidates = []
            for n in range(2, MAX_N + 1):
                if n in selected_set and n != removed:
                    continue
                if n == removed:
                    continue
                # Check coprime to all in temp_selected
                ok = True
                for s in temp_selected:
                    if gcd(n, s) != 1:
                        ok = False
                        break
                if ok:
                    candidates.append(n)
            
            # Now try to find two numbers from candidates that are coprime to each other
            # and whose sum is greater than removed
            best_pair = None
            best_sum = removed
            for a_idx in range(len(candidates)):
                a = candidates[a_idx]
                for b_idx in range(a_idx + 1, len(candidates)):
                    b = candidates[b_idx]
                    if gcd(a, b) == 1:
                        pair_sum = a + b
                        if pair_sum > best_sum:
                            best_sum = pair_sum
                            best_pair = (a, b)
            
            if best_pair is not None:
                # Apply the improvement
                new_selected = temp_selected + list(best_pair)
                # Verify all pairs are coprime
                valid = True
                for i2 in range(len(new_selected)):
                    for j2 in range(i2 + 1, len(new_selected)):
                        if gcd(new_selected[i2], new_selected[j2]) != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    selected = new_selected
                    selected_set = set(selected)
                    improved = True
                    break
    
    # Ensure we return a list of Python ints
    result = [int(x) for x in selected]
    return result
