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
    
    # Map small primes to indices for bitmasking
    p_index = {p: i for i, p in enumerate(small_primes)}
    n_small = len(small_primes)
    
    # Generate all valid subsets of small primes
    # A subset is valid if the product of its elements is <= 256
    # We represent subsets as bitmasks
    valid_subsets = []
    subset_values = []
    
    # Precompute powers to find max value for a given set of distinct primes
    # For a set of distinct primes S, max_val = max { prod(p^e) <= 256 }
    
    def max_value_for_primes(prime_list):
        if not prime_list:
            return 1
        # Recursive or iterative search for max product <= 256
        # Since the number of primes in a valid subset is small (max 3-4), brute force is fine
        # Or simply: start with prod=1, multiply by primes, try powers
        # Better: DFS
        max_val = 1
        def dfs(idx, current_prod):
            nonlocal max_val
            if current_prod > MAX_N:
                return
            if idx == len(prime_list):
                if current_prod > max_val:
                    max_val = current_prod
                return
            p = prime_list[idx]
            # Try powers of p
            power = 1
            while current_prod * power <= MAX_N:
                dfs(idx + 1, current_prod * power)
                power *= p
                if power > MAX_N:
                    break
            # Also skip this prime? No, the subset is fixed. We must use all primes in the subset.
            # The definition of "value of subset S" implies we use ALL primes in S.
            # So we don't skip.
        dfs(0, 1)
        return max_val

    # Iterate over all subsets of small_primes
    # There are 2^31 subsets, which is too many to iterate naively if we check product each time.
    # But we can prune: if product > 256, all supersets are invalid.
    # Use recursive generation with pruning.
    
    def generate_subsets(idx, current_prod, current_mask):
        if time.time() > deadline - 10:
            return
        if idx == n_small:
            if current_prod > 1: # Exclude empty set
                valid_subsets.append(current_mask)
                # Calculate value
                # Extract primes from mask
                ps = [small_primes[i] for i in range(n_small) if (current_mask & (1 << i))]
                val = max_value_for_primes(ps)
                subset_values.append(val)
            return
        
        # Option 1: Do not include small_primes[idx]
        generate_subsets(idx + 1, current_prod, current_mask)
        
        # Option 2: Include small_primes[idx]
        new_prod = current_prod * small_primes[idx]
        if new_prod <= MAX_N:
            generate_subsets(idx + 1, new_prod, current_mask | (1 << idx))
        # If new_prod > MAX_N, we cannot include this prime in any valid subset with the current prefix.
        # Since primes are sorted, adding more primes will only increase the product.
        # So we don't need to go deeper if we include it.
        # But we already handled "do not include".
        # If we include it and it's too big, we stop this branch.

    generate_subsets(0, 1, 0)
    
    if not valid_subsets:
        # Fallback: just use primes individually
        result_numbers.extend(small_primes)
        return [int(x) for x in result_numbers]
    
    n_subsets = len(valid_subsets)
    
    # ILP Formulation
    # Variables x_j for j in range(n_subsets)
    # Maximize sum(subset_values[j] * x_j)
    # Subject to: for each small prime i, sum(x_j for j where mask_j has bit i) <= 1
    # And we want to use ALL small primes? 
    # As argued, using a prime as singleton is always >= not using it.
    # So constraint should be == 1.
    
    c = np.array(subset_values, dtype=float)
    
    # A matrix: n_small x n_subsets
    A = np.zeros((n_small, n_subsets), dtype=float)
    for j, mask in enumerate(valid_subsets):
        for i in range(n_small):
            if mask & (1 << i):
                A[i, j] = 1.0
    
    # Constraints: A @ x == 1
    # Since it's a partition, every prime must be covered exactly once.
    
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
        # Fallback: greedy or simple assignment
        # Assign each small prime to itself
        result_numbers.extend(small_primes)
        
    # Final check and cleanup
    # Ensure distinctness and coprimality
    # If there's a conflict, fall back to primes
    def is_valid(s):
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                if gcd(s[i], s[j]) != 1:
                    return False
        return True
    
    if not is_valid(result_numbers):
        # Fallback to all primes + 1
        result_numbers = primes + [1]
        
    # Ensure types are Python int
    return [int(x) for x in result_numbers]

# EVOLVE-BLOCK-END
