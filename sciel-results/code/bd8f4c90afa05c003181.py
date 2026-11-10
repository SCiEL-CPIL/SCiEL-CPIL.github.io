"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial, gcd
from itertools import combinations
from functools import lru_cache

def _factorize(n):
    """Return prime factorization as dict {prime: exponent}."""
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

def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    
    n = 12
    target = factorial(n)
    target_factors = _factorize(target)
    
    # Generate all divisors of 12! that are > 12
    # First, get all prime factors and their max exponents
    primes = sorted(target_factors.keys())
    max_exps = [target_factors[p] for p in primes]
    
    # Generate all divisors > 12
    def gen_divisors(idx, current):
        if idx == len(primes):
            if current > n:
                yield current
            return
        p = primes[idx]
        e = 0
        while e <= max_exps[idx]:
            yield from gen_divisors(idx + 1, current)
            current *= p
            e += 1
            if current > target:
                break
    
    all_divisors = list(gen_divisors(0, 1))
    all_divisors = sorted(set(all_divisors))
    
    # For each divisor, compute its exponent vector
    div_exp_vectors = {}
    for d in all_divisors:
        div_exp_vectors[d] = _factorize(d)
    
    # Target exponent vector as tuple
    target_vec = tuple(target_factors.get(p, 0) for p in primes)
    
    # We want to find a subset of distinct divisors (each > 12) whose product is exactly 12!
    # i.e., the sum of their exponent vectors equals the target vector
    # Minimize the maximum factor in the subset
    
    # Strategy: Binary search on the maximum allowed factor M
    # For a given M, check if there exists a valid decomposition using only factors <= M
    
    # Pre-sort divisors by value
    sorted_divs = sorted(all_divisors)
    
    # For efficiency, represent exponent vectors as tuples for hashing
    # Use DFS with memoization to check if we can reach target using available divisors
    
    def can_decompose(max_val, divs, target_vec, primes):
        """Check if target can be formed using distinct divisors from divs, all <= max_val."""
        if not divs:
            return target_vec == (0,) * len(primes)
        
        # If target is zero, we're done
        if all(x == 0 for x in target_vec):
            return True
        
        # Pruning: if any component of target is negative, invalid
        if any(x < 0 for x in target_vec):
            return False
        
        # Try each divisor
        for i, d in enumerate(divs):
            if d > max_val:
                break
            dv = _factorize(d)
            dvec = tuple(dv.get(p, 0) for p in primes)
            
            # Check if this divisor fits in remaining target
            if any(dvec[j] > target_vec[j] for j in range(len(primes))):
                continue
            
            new_target = tuple(target_vec[j] - dvec[j] for j in range(len(primes)))
            
            # Recurse with remaining divisors (to ensure distinctness)
            remaining = divs[i+1:]
            if can_decompose(max_val, remaining, new_target, primes):
                return True
        
        return False
    
    # This DFS might be too slow. Let's use a different approach:
    # Meet-in-the-middle or iterative deepening on the number of factors.
    
    # Alternative: Since we want to minimize the max factor, let's try small M values first.
    # The minimum possible max factor is at least ceil((12!)^(1/12)) since we need at most 12 factors.
    # 12! = 479001600
    # (12!)^(1/12) ≈ 479001600^(1/12) ≈ 9.8, so max factor must be at least 10, but > 12 so at least 13.
    
    # Let's try to find the optimal solution by searching M from low to high.
    
    # Actually, let's use a smarter DFS with memoization on the remaining target vector.
    # The state is (index in sorted divisors, remaining target vector)
    
    best_result = None
    best_max = float('inf')
    
    # Try increasing max values
    # The divisors are sorted, so we can binary search on M
    # But first, let's try a direct DFS that minimizes the max factor
    
    # Use DFS that tries to build a decomposition, keeping track of the max factor used
    # and pruning if current max exceeds best_max
    
    divs_sorted = sorted(all_divisors)
    
    @lru_cache(maxsize=None)
    def dfs(idx, remaining_vec):
        """
        Try to form remaining_vec using divisors from index idx onwards.
        Returns the minimum possible maximum factor in the subset, or None if impossible.
        """
        if all(x == 0 for x in remaining_vec):
            return 0  # No more factors needed, max is 0
        
        if idx >= len(divs_sorted):
            return None
        
        best = None
        
        # Option 1: Skip this divisor
        result_skip = dfs(idx + 1, remaining_vec)
        if result_skip is not None:
            best = result_skip
        
        # Option 2: Use this divisor
        d = divs_sorted[idx]
        dv = _factorize(d)
        dvec = tuple(dv.get(p, 0) for p in primes)
        
        # Check if divisor fits
        if all(dvec[j] <= remaining_vec[j] for j in range(len(primes))):
            new_remaining = tuple(remaining_vec[j] - dvec[j] for j in range(len(primes)))
            result_use = dfs(idx + 1, new_remaining)
            if result_use is not None:
                candidate = max(d, result_use)
                if best is None or candidate < best:
                    best = candidate
        
        return best
    
    # The DFS above may have too many states. Let's limit the search.
    # Actually, the number of divisors of 12! is not too large.
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # Number of divisors = (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 11*6*3*2*2 = 792
    # But we only consider divisors > 12, so maybe ~700.
    
    # The DFS with memoization on (idx, remaining_vec) should work.
    # remaining_vec has 5 components, each bounded by the target exponents.
    # Max states: 792 * 11 * 6 * 3 * 2 * 2 = 792 * 792 = ~627k, which is manageable.
    
    try:
        result = dfs(0, target_vec)
        if result is not None:
            # We found the optimal max factor value, now we need to reconstruct the actual factors
            # Let's modify the DFS to also return the factors used
            
            @lru_cache(maxsize=None)
            def dfs_reconstruct(idx, remaining_vec):
                """
                Returns (max_factor, tuple_of_factors) or None.
                """
                if all(x == 0 for x in remaining_vec):
                    return (0, ())
                
                if idx >= len(divs_sorted):
                    return None
                
                best = None
                
                # Option 1: Skip
                result_skip = dfs_reconstruct(idx + 1, remaining_vec)
                if result_skip is not None:
                    best = result_skip
                
                # Option 2: Use
                d = divs_sorted[idx]
                dv = _factorize(d)
                dvec = tuple(dv.get(p, 0) for p in primes)
                
                if all(dvec[j] <= remaining_vec[j] for j in range(len(primes))):
                    new_remaining = tuple(remaining_vec[j] - dvec[j] for j in range(len(primes)))
                    result_use = dfs_reconstruct(idx + 1, new_remaining)
                    if result_use is not None:
                        max_f, factors = result_use
                        candidate_max = max(d, max_f)
                        candidate_factors = (d,) + factors
                        if best is None or candidate_max < best[0]:
                            best = (candidate_max, candidate_factors)
                
                return best
            
            result = dfs_reconstruct(0, target_vec)
            if result is not None:
                max_f, factors = result
                return list(factors)
    except Exception:
        pass
    
    # Fallback: return the single factor 12!
    return [factorial(12)]

# EVOLVE-BLOCK-END
