"""CSP-style search with prime-exponent vectors, forward checking, and M-iteration."""
# EVOLVE-BLOCK-START
import time
from math import factorial, isqrt
from functools import lru_cache

def construct():
    start_time = time.time()
    timeout = 55.0  # safety margin before 90s kill

    target = factorial(12)  # 479001600

    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    def prime_factorization(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    target_pf = prime_factorization(target)
    primes = sorted(target_pf.keys())
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    n_primes = len(primes)

    def pf_to_vec(pf):
        vec = [0] * n_primes
        for p, e in pf.items():
            vec[prime_to_idx[p]] = e
        return tuple(vec)

    target_vec = pf_to_vec(target_pf)

    # Precompute: for each divisor d > 12 of target, store its exponent vector
    def get_divisors(n):
        divs = []
        for i in range(1, isqrt(n) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return sorted(divs)

    all_divs = get_divisors(target)
    valid_divs = [d for d in all_divs if d > 12]

    # Map each valid divisor to its exponent vector
    divisor_vecs = {}
    for d in valid_divs:
        divisor_vecs[d] = pf_to_vec(prime_factorization(d))

    def can_divide(rem_vec, div_vec):
        """Check if div_vec divides rem_vec (component-wise)."""
        for i in range(n_primes):
            if rem_vec[i] < div_vec[i]:
                return False
        return True

    def sub_vec(rem_vec, div_vec):
        """Subtract div_vec from rem_vec."""
        return tuple(rem_vec[i] - div_vec[i] for i in range(n_primes))

    def is_zero_vec(vec):
        return all(v == 0 for v in vec)

    n_valid = len(valid_divs)

    # Binary search on M: find the smallest M such that a valid decomposition
    # exists using only factors <= M.
    
    def feasible(M_idx):
        """Check if a valid decomposition exists using valid_divs[0..M_idx]."""
        candidates = valid_divs[:M_idx + 1]
        candidate_vecs = [divisor_vecs[c] for c in candidates]
        n_cand = len(candidates)
        
        # Precompute suffix sums for forward checking
        # suffix_sums[j][i] = sum of candidate_vecs[k][i] for k >= j
        suffix_sums = [[0] * n_primes for _ in range(n_cand + 1)]
        for j in range(n_cand - 1, -1, -1):
            for i in range(n_primes):
                suffix_sums[j][i] = suffix_sums[j + 1][i] + candidate_vecs[j][i]
        
        memo = {}
        
        def dfs(start_idx, rem_vec, max_factors_left):
            if time.time() - start_time > timeout:
                return False
            
            if is_zero_vec(rem_vec):
                return max_factors_left >= 1
            
            if max_factors_left <= 0:
                return False
            
            key = (start_idx, rem_vec, max_factors_left)
            if key in memo:
                return memo[key]
            
            # Forward checking: for each prime, remaining exponent must be <= suffix sum
            for i in range(n_primes):
                if rem_vec[i] > suffix_sums[start_idx][i]:
                    memo[key] = False
                    return False
            
            for idx in range(start_idx, n_cand):
                cv = candidate_vecs[idx]
                if not can_divide(rem_vec, cv):
                    continue
                
                new_rem = sub_vec(rem_vec, cv)
                if dfs(idx + 1, new_rem, max_factors_left - 1):
                    memo[key] = True
                    return True
            
            memo[key] = False
            return False
        
        return dfs(0, target_vec, 12)
    
    # Binary search for the smallest M_idx where feasible(M_idx) is True
    lo, hi = 0, n_valid - 1
    best_idx = n_valid - 1
    
    while lo <= hi:
        if time.time() - start_time > timeout:
            break
        mid = (lo + hi) // 2
        if feasible(mid):
            best_idx = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    # Now extract the actual solution using the best M
    M = valid_divs[best_idx]
    candidates = valid_divs[:best_idx + 1]
    candidate_vecs = [divisor_vecs[c] for c in candidates]
    n_cand = len(candidates)
    
    # Precompute suffix sums
    suffix_sums = [[0] * n_primes for _ in range(n_cand + 1)]
    for j in range(n_cand - 1, -1, -1):
        for i in range(n_primes):
            suffix_sums[j][i] = suffix_sums[j + 1][i] + candidate_vecs[j][i]
    
    memo = {}
    
    def dfs_extract(start_idx, rem_vec, max_factors_left, current_factors):
        if time.time() - start_time > timeout:
            return None
        
        if is_zero_vec(rem_vec):
            if max_factors_left >= 1:
                return list(current_factors)
            return None
        
        if max_factors_left <= 0:
            return None
        
        key = (start_idx, rem_vec, max_factors_left)
        if key in memo:
            if memo[key] is None:
                return None
        
        # Forward checking
        for i in range(n_primes):
            if rem_vec[i] > suffix_sums[start_idx][i]:
                memo[key] = None
                return None
        
        for idx in range(start_idx, n_cand):
            cv = candidate_vecs[idx]
            if not can_divide(rem_vec, cv):
                continue
            
            new_rem = sub_vec(rem_vec, cv)
            current_factors.append(candidates[idx])
            result = dfs_extract(idx + 1, new_rem, max_factors_left - 1, current_factors)
            current_factors.pop()
            
            if result is not None:
                memo[key] = result
                return result
        
        memo[key] = None
        return None
    
    solution = dfs_extract(0, target_vec, 12, [])
    
    if solution is not None:
        return solution
    
    # Fallback: return [target]
    return [target]

# EVOLVE-BLOCK-END
