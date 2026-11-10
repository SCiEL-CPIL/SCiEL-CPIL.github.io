# EVOLVE-BLOCK-START
import time
from math import factorial
from functools import lru_cache

TARGET = factorial(12)  # 479001600

def _get_all_divisors():
    """Generate all divisors of TARGET, sorted."""
    primes = [2, 3, 5, 7, 11]
    exps = [10, 5, 2, 1, 1]
    divisors = [1]
    for p, e in zip(primes, exps):
        new_divs = []
        for d in divisors:
            for i in range(e + 1):
                new_divs.append(d * (p ** i))
        divisors = new_divs
    divisors.sort()
    return divisors

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # 1. Get all divisors of 12! that are > 12
    all_divs = _get_all_divisors()
    valid_divs = [d for d in all_divs if d > 12]
    n_valid = len(valid_divs)
    
    if n_valid == 0:
        return [TARGET]

    # Precompute suffix products of the largest 7 elements for pruning
    # suffix_max_prod[i] = product of the largest min(7, len(valid_divs)-i) elements in valid_divs[i:]
    # We only need this for the current candidate set in DFS, but since we binary search,
    # the candidate set changes. 
    # However, the pruning check is cheap enough to compute on the fly or we can pass the list.
    # Let's just compute it on the fly in DFS to avoid complex precomputation for varying prefixes.
    
    def is_feasible(hi_idx):
        # We check if a subset of valid_divs[0..hi_idx] multiplies to TARGET.
        # We use DFS with pruning.
        cands = valid_divs[:hi_idx + 1]
        n_cands = len(cands)
        
        # If the product of the largest 7 elements in cands is < TARGET, infeasible.
        # This is a global check for this hi_idx.
        largest_7 = cands[-7:] if n_cands >= 7 else cands
        max_possible = 1
        for v in largest_7:
            max_possible *= v
        if max_possible < TARGET:
            return False
            
        @lru_cache(maxsize=None)
        def dfs(rem, start_idx):
            if rem == 1:
                return True
            if start_idx >= n_cands:
                return False
            
            # Pruning: Check if product of largest available elements is enough
            # Largest available are at indices [n_cands-7 ... n_cands-1] intersected with [start_idx ... n_cands-1]
            # Since we are picking from start_idx onwards, the largest available are the last 7 of cands[start_idx:]
            # Which are cands[max(start_idx, n_cands-7) : n_cands]
            start_check = max(start_idx, n_cands - 7)
            prod = 1
            for j in range(start_check, n_cands):
                prod *= cands[j]
            if prod < rem:
                return False

            # Iterate from largest to smallest to reduce rem quickly
            # Find the largest index i >= start_idx such that cands[i] <= rem
            # Since cands is sorted, we can binary search or just iterate down.
            # Iterating down is fine.
            for i in range(n_cands - 1, start_idx - 1, -1):
                if cands[i] > rem:
                    continue
                if rem % cands[i] == 0:
                    if dfs(rem // cands[i], i + 1):
                        return True
            return False

        res = dfs(TARGET, 0)
        dfs.cache_clear()
        return res

    # 2. Binary search for the smallest maximum factor B
    lo = 0
    hi = n_valid - 1
    best_hi = -1
    
    while lo <= hi:
        mid = (lo + hi) // 2
        if is_feasible(mid):
            best_hi = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
        if time.time() > deadline - 5.0:
            break

    if best_hi == -1:
        return [TARGET]

    # 3. Retrieve the factors
    def get_factors(hi_idx):
        cands = valid_divs[:hi_idx + 1]
        n_cands = len(cands)
        
        @lru_cache(maxsize=None)
        def dfs(rem, start_idx):
            if rem == 1:
                return []
            if start_idx >= n_cands:
                return None
            
            # Pruning
            start_check = max(start_idx, n_cands - 7)
            prod = 1
            for j in range(start_check, n_cands):
                prod *= cands[j]
            if prod < rem:
                return None

            for i in range(n_cands - 1, start_idx - 1, -1):
                if cands[i] > rem:
                    continue
                if rem % cands[i] == 0:
                    sub = dfs(rem // cands[i], i + 1)
                    if sub is not None:
                        return [cands[i]] + sub
            return None
        
        res = dfs(TARGET, 0)
        dfs.cache_clear()
        return res

    factors = get_factors(best_hi)
    
    if factors is None:
        return [TARGET]
        
    return factors

# EVOLVE-BLOCK-END
