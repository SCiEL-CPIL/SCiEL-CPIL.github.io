import math
import time
import heapq

def construct():
    T12 = math.factorial(12)
    start_time = time.time()
    deadline = start_time + 85

    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    # Generate all divisors of 12! that are > 12
    all_divisors = []
    
    def generate_divisors(idx, current_val):
        if idx == len(primes):
            if current_val > 12:
                all_divisors.append(current_val)
            return
        p = primes[idx]
        e = exponents[idx]
        val = 1
        for i in range(e + 1):
            generate_divisors(idx + 1, current_val * val)
            val *= p

    generate_divisors(0, 1)
    all_divisors.sort()
    
    # Map each divisor to its prime exponent vector
    divisor_to_exp = {}
    for d in all_divisors:
        temp = d
        exp_vec = []
        for p in primes:
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            exp_vec.append(count)
        divisor_to_exp[d] = tuple(exp_vec)
    
    # State space: all possible remaining exponent vectors
    # Each state is a tuple (e2, e3, e5, e7, e11) where 0 <= e_i <= exponents[i]
    # Start state: full exponents, end state: all zeros
    
    # Dijkstra's algorithm on the DAG of exponent vectors
    # Edge from state u to state v if v = u - exp_vec(d) for some divisor d
    # Edge weight: max(current_max, d) where d is the divisor used
    
    # We want to minimize the largest factor used
    
    # Precompute all valid states and transitions
    from functools import lru_cache
    
    # For each state, find all divisors that can be applied
    # A divisor d can be applied to state s if exp_vec(d)[i] <= s[i] for all i
    
    # Use Dijkstra with state = (exponent_vector, current_max_factor)
    # But we need to track the path to reconstruct the solution
    
    # Actually, let's use a different approach:
    # Binary search on M (the maximum allowed factor)
    # For a given M, check if we can decompose 12! into distinct factors all <= M and > 12
    
    def can_decompose(M):
        """Check if 12! can be decomposed into distinct factors all in (12, M]"""
        valid_divisors = [d for d in all_divisors if d <= M]
        n = len(valid_divisors)
        
        if n == 0:
            return None, None
        
        # Use DFS with memoization on (remaining_product, min_idx)
        # But remaining_product can be large, so use exponent vector instead
        
        # State: (remaining_exp_vector, min_idx)
        # remaining_exp_vector is a tuple of 5 integers
        
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dfs(rem_exp, min_idx):
            """
            rem_exp: tuple of 5 ints, remaining prime exponents
            min_idx: minimum index in valid_divisors to use
            Returns: list of divisors used, or None
            """
            if time.time() > deadline:
                return None
            
            # Check if we're done
            if all(e == 0 for e in rem_exp):
                return []
            
            if min_idx >= n:
                return None
            
            # Try each divisor from min_idx onwards
            for i in range(min_idx, n):
                d = valid_divisors[i]
                exp_vec = divisor_to_exp[d]
                
                # Check if we can apply this divisor
                can_apply = True
                new_rem_exp = []
                for j in range(5):
                    if rem_exp[j] < exp_vec[j]:
                        can_apply = False
                        break
                    new_rem_exp.append(rem_exp[j] - exp_vec[j])
                
                if not can_apply:
                    continue
                
                new_rem_exp = tuple(new_rem_exp)
                result = dfs(new_rem_exp, i + 1)
                if result is not None:
                    return [d] + result
            
            return None
        
        start_exp = tuple(exponents)
        result = dfs(start_exp, 0)
        if result is not None:
            return result, M
        return None, None
    
    # Binary search on M
    lo, hi = 13, 25000
    best_solution = None
    best_M = None
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        sol, M_val = can_decompose(mid)
        if sol is not None:
            best_solution = sol
            best_M = M_val
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: try direct DFS without binary search
    def direct_dfs():
        valid_divisors = all_divisors
        n = len(valid_divisors)
        
        @lru_cache(maxsize=None)
        def dfs(rem_exp, min_idx):
            if time.time() > deadline:
                return None
            
            if all(e == 0 for e in rem_exp):
                return []
            
            if min_idx >= n:
                return None
            
            for i in range(min_idx, n):
                d = valid_divisors[i]
                exp_vec = divisor_to_exp[d]
                
                can_apply = True
                new_rem_exp = []
                for j in range(5):
                    if rem_exp[j] < exp_vec[j]:
                        can_apply = False
                        break
                    new_rem_exp.append(rem_exp[j] - exp_vec[j])
                
                if not can_apply:
                    continue
                
                new_rem_exp = tuple(new_rem_exp)
                result = dfs(new_rem_exp, i + 1)
                if result is not None:
                    return [d] + result
            
            return None
        
        return dfs(tuple(exponents), 0)
    
    sol = direct_dfs()
    if sol is not None:
        return sol
    
    return [T12]
