import time
from math import gcd, isqrt
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s limit, leaving margin before 90s kill
    
    MAX_N = 256
    
    # Get primes
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(MAX_N)
    
    # Separate primes
    small_primes = [p for p in primes if p <= 128]
    large_primes = [p for p in primes if p > 128]
    
    # 1 is always included
    result_numbers = [1]
    result_numbers.extend(large_primes)
    
    # Map small primes to indices
    p_index = {p: i for i, p in enumerate(small_primes)}
    n_small = len(small_primes)
    
    # For each small prime, find the maximum power p^k <= 256
    # This is the value if used alone
    def max_power_value(p):
        val = 1
        while val * p <= MAX_N:
            val *= p
        return val
    
    # Precompute max power value for each small prime
    prime_max_vals = [max_power_value(p) for p in small_primes]
    
    # Generate all valid subsets of small primes
    # A subset is valid if there exists an assignment of powers such that product <= 256
    # For a set of distinct primes S, the max value is max { prod(p_i^e_i) <= 256 }
    
    def max_value_for_primes(prime_list):
        """Find maximum product of prime powers <= MAX_N for given distinct primes."""
        if not prime_list:
            return 1
        max_val = 1
        n = len(prime_list)
        
        def dfs(idx, current_prod):
            nonlocal max_val
            if current_prod > MAX_N:
                return
            if idx == n:
                if current_prod > max_val:
                    max_val = current_prod
                return
            p = prime_list[idx]
            power = 1
            while current_prod * power <= MAX_N:
                dfs(idx + 1, current_prod * power)
                power *= p
                if power > MAX_N:
                    break
        
        dfs(0, 1)
        return max_val
    
    # Generate valid subsets using recursive generation with pruning
    valid_subsets = []
    subset_values = []
    subset_primes = []  # Store the actual primes in each subset
    
    def generate_subsets(idx, current_prod, current_mask, current_primes):
        if time.time() > deadline - 15:
            return
        if idx == n_small:
            if current_prod > 1:  # Exclude empty set
                valid_subsets.append(current_mask)
                ps = list(current_primes)
                val = max_value_for_primes(ps)
                subset_values.append(val)
                subset_primes.append(ps)
            return
        
        p = small_primes[idx]
        
        # Option 1: Do not include small_primes[idx]
        generate_subsets(idx + 1, current_prod, current_mask, current_primes)
        
        # Option 2: Include small_primes[idx]
        new_prod = current_prod * p
        if new_prod <= MAX_N:
            generate_subsets(idx + 1, new_prod, current_mask | (1 << idx), current_primes + [p])
    
    generate_subsets(0, 1, 0, [])
    
    if not valid_subsets:
        # Fallback: just use primes individually
        result_numbers.extend(small_primes)
        return [int(x) for x in result_numbers]
    
    n_subsets = len(valid_subsets)
    
    # ILP Formulation
    # Variables x_j for j in range(n_subsets)
    # Maximize sum(subset_values[j] * x_j)
    # Subject to: for each small prime i, sum(x_j for j where mask_j has bit i) == 1
    # (Each prime must be used exactly once)
    
    c = np.array(subset_values, dtype=float)
    
    # A matrix: n_small x n_subsets
    A = np.zeros((n_small, n_subsets), dtype=float)
    for j, mask in enumerate(valid_subsets):
        for i in range(n_small):
            if mask & (1 << i):
                A[i, j] = 1.0
    
    # Constraints: A @ x == 1
    constraints = LinearConstraint(A, 1, 1)
    integrality = np.ones(n_subsets)
    bounds = Bounds(0, 1)
    
    res = milp(c=-c, constraints=constraints, integrality=integrality, bounds=bounds)
    
    if res.status == 0:
        # res.x contains the optimal solution
        selected_masks = []
        for j in range(n_subsets):
            if res.x[j] > 0.5:
                selected_masks.append(valid_subsets[j])
        
        # Construct numbers from selected masks
        for mask in selected_masks:
            ps = [small_primes[i] for i in range(n_small) if (mask & (1 << i))]
            val = max_value_for_primes(ps)
            result_numbers.append(val)
    else:
        # Fallback: greedy merge
        slots = list(small_primes)
        while True:
            if time.time() > deadline - 5:
                break
            best_gain = 0
            best_i = -1
            best_j = -1
            n = len(slots)
            for i in range(n):
                for j in range(i + 1, n):
                    merged = slots[i] * slots[j]
                    if merged > MAX_N:
                        continue
                    gain = merged - slots[i] - slots[j]
                    if gain > best_gain:
                        best_gain = gain
                        best_i = i
                        best_j = j
            if best_i == -1 or best_gain <= 0:
                break
            slots[best_i] = slots[best_i] * slots[best_j]
            slots.pop(best_j)
        result_numbers.extend(slots)
    
    # Local search refinement: try swapping assignments
    # Check if we can improve by trying different power assignments
    def try_improve(result_nums):
        """Try to improve the solution by local search."""
        if time.time() > deadline - 5:
            return result_nums
        
        # For each number that is a product of multiple primes, try different power splits
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(result_nums)):
                n = result_nums[i]
                # Factorize n
                factors = []
                temp = n
                for p in small_primes:
                    while temp % p == 0:
                        factors.append(p)
                        temp //= p
                if temp > 1:
                    factors.append(temp)
                
                if len(factors) <= 1:
                    continue
                
                # Try to redistribute: split n into two coprime factors a, b > 1
                # where a*b = n, and check if a + b > n (shouldn't happen since n >= a*b for a,b >= 2)
                # Actually we want to see if we can replace n with a different combination
                # This is complex, so let's try a simpler approach:
                # For each prime factor p of n, try reducing its power and see if we can
                # increase another number's value
                
                # Simpler: just verify validity and move on
                pass
        
        return result_nums
    
    result_numbers = try_improve(result_numbers)
    
    # Final check and cleanup
    def is_valid(s):
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if gcd(s[i], s[j]) != 1:
                    return False
        return True
    
    if not is_valid(result_numbers):
        # Fallback to all primes + 1
        result_numbers = primes + [1]
    
    # Ensure distinctness
    seen = set()
    unique_result = []
    for x in result_numbers:
        if x not in seen:
            seen.add(x)
            unique_result.append(x)
    result_numbers = unique_result
    
    # Ensure types are Python int
    return [int(x) for x in result_numbers]

# EVOLVE-BLOCK-END
