import time
from math import factorial
from functools import lru_cache
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    N = 479001600  # 12!
    
    # Prime factorization of 12!
    # 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    target_exp = [10, 5, 2, 1, 1]
    target_tuple = tuple(target_exp)
    
    # Generate all divisors of N
    # A divisor is defined by exponents e_i in [0, target_exp[i]]
    divisors = []
    
    def gen_divisors(idx, current_exp, current_val):
        if idx == len(primes):
            if current_val > 12:
                divisors.append((current_val, tuple(current_exp)))
            return
        for e in range(target_exp[idx] + 1):
            current_exp.append(e)
            gen_divisors(idx + 1, current_exp, current_val * (primes[idx] ** e))
            current_exp.pop()
            
    gen_divisors(0, [], 1)
    
    # Sort divisors by value
    divisors.sort(key=lambda x: x[0])
    
    n_divs = len(divisors)
    
    # We want to find the smallest M such that there exists a subset of distinct divisors
    # all <= M, whose product is N.
    
    # Precompute suffix sums of exponents for pruning
    # suffix_exp_sum[i] = sum of exponents of divisors[i:]
    # This helps prune if the remaining divisors can't cover the remaining exponents.
    suffix_exp_sum = [None] * (n_divs + 1)
    suffix_exp_sum[n_divs] = [0] * 5
    for i in range(n_divs - 1, -1, -1):
        prev = suffix_exp_sum[i+1]
        curr_exp = divisors[i][1]
        suffix_exp_sum[i] = [prev[j] + curr_exp[j] for j in range(5)]
        
    # We will use a DFS that picks factors in increasing order of value.
    # To minimize the largest factor, we can iterate through candidate max factors M.
    # However, a smarter approach is to just find ANY valid set first, then try to improve it?
    # No, the objective is to minimize the max factor.
    # Let's do a binary search on the max factor value M.
    # For a given M, we filter divisors <= M.
    # Then we check if we can form N from a subset of these.
    
    # Optimization:
    # The number of divisors is ~800.
    # The state space for DFS is (remaining_exponents, last_index).
    # remaining_exponents is a tuple of 5 ints.
    # The number of possible remaining exponent tuples is (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 11*6*3*2*2 = 792.
    # The last_index is 0..n_divs.
    # So the state space is small enough for memoization.
    
    # We need to be careful with the "distinct" constraint.
    # By picking indices in increasing order (start_idx), we ensure distinctness.
    
    def check_validity(max_val, time_limit):
        """
        Check if a valid set of distinct divisors <= max_val exists.
        Returns the set of factors if found, else None.
        """
        # Filter divisors
        valid_divs = [d for d in divisors if d[0] <= max_val]
        if not valid_divs:
            return None
            
        n_v = len(valid_divs)
        
        # Recompute suffix sums for this subset
        suf = [None] * (n_v + 1)
        suf[n_v] = [0] * 5
        for i in range(n_v - 1, -1, -1):
            prev = suf[i+1]
            curr_exp = valid_divs[i][1]
            suf[i] = [prev[j] + curr_exp[j] for j in range(5)]
            
        # Check if total exponents in valid_divs can cover target
        for j in range(5):
            if suf[0][j] < target_exp[j]:
                return None

        # DFS with memoization
        # State: (rem_exp_tuple, start_idx)
        # We use a dictionary for memo to allow different valid_divs lists
        memo = {}
        
        def dfs(rem_exp_tuple, start_idx):
            key = (rem_exp_tuple, start_idx)
            if key in memo:
                return memo[key]
                
            rem_exp = list(rem_exp_tuple)
            if all(e == 0 for e in rem_exp):
                memo[key] = [] # Empty list of factors needed
                return []
                
            # Pruning
            if start_idx >= n_v:
                memo[key] = None
                return None
                
            suffix = suf[start_idx]
            for j in range(5):
                if suffix[j] < rem_exp[j]:
                    memo[key] = None
                    return None
                    
            # Try picking a divisor from start_idx onwards
            # We can also prune if the number of factors chosen so far + remaining needed > 12?
            # We don't track count in state, but we can add it if needed.
            # For now, 12 is a loose constraint.
            
            for i in range(start_idx, n_v):
                val, exp = valid_divs[i]
                fits = True
                for j in range(5):
                    if exp[j] > rem_exp[j]:
                        fits = False
                        break
                if fits:
                    new_rem = tuple(rem_exp[j] - exp[j] for j in range(5))
                    res = dfs(new_rem, i + 1)
                    if res is not None:
                        # Prepend current factor
                        result = [val] + res
                        memo[key] = result
                        return result
                        
            memo[key] = None
            return None
            
        return dfs(tuple(target_exp), 0)

    # Binary search on M
    # unique_values are the sorted divisor values
    unique_vals = sorted(set(d[0] for d in divisors))
    
    best_result = None
    low = 0
    high = len(unique_vals) - 1
    
    # We must ensure at least one factor is found.
    # The trivial solution is [N]. N is in unique_vals.
    
    while low <= high:
        if time.time() > deadline:
            break
            
        mid = (low + high) // 2
        M = unique_vals[mid]
        
        res = check_validity(M, deadline)
        
        if res is not None:
            best_result = res
            high = mid - 1
        else:
            low = mid + 1
            
    if best_result is None:
        # Fallback to trivial solution
        best_result = [N]
        
    return best_result
