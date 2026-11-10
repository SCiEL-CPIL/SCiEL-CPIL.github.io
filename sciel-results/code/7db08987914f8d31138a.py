"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial
from functools import lru_cache

TARGET = factorial(12)  # 479001600
# Prime factorization: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
PRIMES = [2, 3, 5, 7, 11]
MAX_EXPS = (10, 5, 2, 1, 1)

def _factor_to_expvec(n):
    """Convert a divisor of TARGET to its prime exponent tuple."""
    vec = []
    for p in PRIMES:
        cnt = 0
        while n % p == 0:
            cnt += 1
            n //= p
        vec.append(cnt)
    return tuple(vec)

def _get_all_divisors():
    """Generate all divisors of TARGET, sorted, with their exponent vectors."""
    primes = [2, 3, 5, 7, 11]
    exps = [10, 5, 2, 1, 1]
    divisors = [(1, (0, 0, 0, 0, 0))]
    for p, e in zip(primes, exps):
        new_divs = []
        for d, vec in divisors:
            for i in range(e + 1):
                new_vec = list(vec)
                new_vec[PRIMES.index(p)] = i
                new_divs.append((d * (p ** i), tuple(new_vec)))
        divisors = new_divs
    divisors.sort(key=lambda x: x[0])
    return divisors

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # 1. Get all divisors of 12! that are > 12, with exponent vectors
    all_divs = _get_all_divisors()
    valid_divs = [(d, vec) for d, vec in all_divs if d > 12]
    n_valid = len(valid_divs)
    target_vec = MAX_EXPS

    # 2. Binary search for the smallest maximum factor B
    # Feasibility: can we pick a subset of valid_divs[0..hi_idx] whose
    # exponent vectors sum to target_vec?
    
    def is_feasible(hi_idx):
        # DP over exponent vectors using a set
        reachable = {target_vec}
        # We work backwards: start from target, subtract each divisor's vec
        # Actually, let's do forward DP: start from (0,0,0,0,0), add divisors
        # But that's expensive. Let's use the "remaining" approach with memoization.
        
        @lru_cache(maxsize=None)
        def dfs(idx, rem_vec):
            if rem_vec == (0, 0, 0, 0, 0):
                return True
            if idx > hi_idx:
                return False
            
            # Option 1: Skip valid_divs[idx]
            if dfs(idx + 1, rem_vec):
                return True
            
            # Option 2: Take valid_divs[idx] if it divides rem (component-wise)
            d, vec = valid_divs[idx]
            can_take = all(rem_vec[i] >= vec[i] for i in range(5))
            if can_take:
                new_rem = tuple(rem_vec[i] - vec[i] for i in range(5))
                if dfs(idx + 1, new_rem):
                    return True
            
            return False
        
        res = dfs(0, target_vec)
        dfs.cache_clear()
        return res

    # Binary search
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
        @lru_cache(maxsize=None)
        def dfs(idx, rem_vec):
            if rem_vec == (0, 0, 0, 0, 0):
                return []
            if idx > hi_idx:
                return None
            
            # Option 1: Skip
            res = dfs(idx + 1, rem_vec)
            if res is not None:
                return res
            
            # Option 2: Take
            d, vec = valid_divs[idx]
            can_take = all(rem_vec[i] >= vec[i] for i in range(5))
            if can_take:
                new_rem = tuple(rem_vec[i] - vec[i] for i in range(5))
                sub = dfs(idx + 1, new_rem)
                if sub is not None:
                    return [d] + sub
            
            return None
        
        res = dfs(0, target_vec)
        dfs.cache_clear()
        return res

    factors = get_factors(best_hi)
    
    if factors is None:
        return [TARGET]
        
    return factors

# EVOLVE-BLOCK-END
