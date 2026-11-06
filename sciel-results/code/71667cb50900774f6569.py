import math
import time
from functools import lru_cache

T12 = math.factorial(12)
PRIMES = [2, 3, 5, 7, 11]
EXP_TOTAL = (10, 5, 2, 1, 1)

def _exp_vector(n):
    v = [0] * 5
    for i, p in enumerate(PRIMES):
        while n % p == 0:
            n //= p
            v[i] += 1
    return tuple(v)

# Precompute all divisors
ALL_DIVS = []
ALL_EXP_VECS = []
def _init_divisors():
    limit = math.isqrt(T12)
    seen = set()
    for i in range(1, limit + 1):
        if T12 % i == 0:
            for d in (i, T12 // i):
                if d > 12 and d not in seen:
                    seen.add(d)
                    ALL_DIVS.append(d)
    ALL_DIVS.sort()
    for d in ALL_DIVS:
        ALL_EXP_VECS.append(_exp_vector(d))

_init_divisors()

def _find_factors(M, deadline):
    """
    Find a set of distinct factors from ALL_DIVS <= M with product T12.
    Returns list of factors or None.
    """
    # Filter divisors
    indices = [i for i, d in enumerate(ALL_DIVS) if d <= M]
    if not indices:
        return None
    
    # Sort indices by divisor value descending? 
    # Trying larger factors first might reduce remaining product faster.
    indices.sort(key=lambda i: ALL_DIVS[i], reverse=True)
    
    n = len(indices)
    
    # Memoization
    # State: (pos, rem_exp)
    # pos is index in the `indices` list
    # rem_exp is tuple of 5 ints
    
    # We want to find ANY solution.
    # To retrieve the solution, we can store the choice (take or skip) in a dict or just re-run with a path tracker.
    # Given small N, we can just use a simple DFS with backtracking and return the first found path.
    
    solution = []
    
    def dfs(pos, rem_exp):
        if time.time() > deadline:
            return False
        
        # Check if all exponents are 0
        if all(e == 0 for e in rem_exp):
            return True
            
        if pos >= n:
            return False
            
        # Pruning: 
        # If the product of remaining exponents is 1, we are done (handled above).
        # If the remaining product is not divisible by any of the remaining divisors? 
        # This is hard to check quickly.
        
        # Option 1: Take divisor at indices[pos]
        i = indices[pos]
        d = ALL_DIVS[i]
        ev = ALL_EXP_VECS[i]
        
        # Check if we can take it
        can_take = True
        new_rem = [0]*5
        for k in range(5):
            if ev[k] > rem_exp[k]:
                can_take = False
                break
            new_rem[k] = rem_exp[k] - ev[k]
            
        if can_take:
            solution.append(d)
            if dfs(pos + 1, tuple(new_rem)):
                return True
            solution.pop()
            
        # Option 2: Skip divisor at indices[pos]
        if dfs(pos + 1, rem_exp):
            return True
            
        return False
        
    if dfs(0, EXP_TOTAL):
        return solution
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # Binary search for minimum M
    lo = 13
    hi = T12
    best_M = T12
    best_factors = [T12] # Fallback
    
    # We can also just iterate M upwards if the range is small, but binary search is safer.
    # However, the feasibility check is not monotonic in a simple way? 
    # If M is feasible, is M+1 feasible? Yes, because the set of allowed divisors grows.
    # So feasibility is monotonic.
    
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        factors = _find_factors(mid, deadline)
        if factors:
            best_M = mid
            best_factors = factors
            hi = mid - 1
        else:
            lo = mid + 1
            
    # Verify best_factors
    if best_factors:
        prod = 1
        for f in best_factors:
            prod *= f
        if prod == T12 and len(set(best_factors)) == len(best_factors) and all(f > 12 for f in best_factors) and 1 <= len(best_factors) <= 12:
            return best_factors
            
    # Fallback
    return [T12]
