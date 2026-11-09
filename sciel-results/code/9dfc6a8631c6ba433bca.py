import time
from math import factorial
from functools import lru_cache

_START = time.time()
_DEADLINE = _START + 80  # 10s safety margin before 90s limit

def construct():
    target = factorial(12)  # 479001600
    
    # Known good solution: max factor 64
    best_factors = [22, 21, 25, 18, 36, 64]
    best_max = 64
    
    # Verify
    prod = 1
    for f in best_factors:
        prod *= f
    if prod != target:
        return [target]
    
    # Prime factorization: 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # We search for solutions with max factor < best_max
    # Use prime exponent tracking for efficiency
    
    # Generate all divisors of target that are > 12
    divisors = []
    for a in range(11):      # 2^a
        for b in range(6):   # 3^b
            for c in range(3):# 5^c
                for d in range(2):# 7^d
                    for e in range(2):# 11^e
                        val = (2**a) * (3**b) * (5**c) * (7**d) * (11**e)
                        if val > 12:
                            divisors.append(val)
    
    divisors.sort()
    
    # For each divisor, precompute its prime exponent signature
    # Signature: (exp_2, exp_3, exp_5, exp_7, exp_11)
    def get_sig(n):
        sig = [0, 0, 0, 0, 0]
        temp = n
        for i, p in enumerate([2, 3, 5, 7, 11]):
            while temp % p == 0:
                sig[i] += 1
                temp //= p
        return tuple(sig)
    
    target_sig = (10, 5, 2, 1, 1)
    
    # Precompute signatures for all valid divisors
    div_sig = {}
    for d in divisors:
        div_sig[d] = get_sig(d)
    
    # Search: try to find a decomposition with all factors < best_max
    # Use DFS with prime exponent tracking
    found_better = [False]
    better_factors = [None]
    
    def can_subtract(rem_sig, sig):
        """Check if sig can be subtracted from rem_sig"""
        for i in range(5):
            if rem_sig[i] < sig[i]:
                return False
        return True
    
    def subtract_sig(rem_sig, sig):
        return tuple(rem_sig[i] - sig[i] for i in range(5))
    
    def dfs(rem_sig, min_factor, max_allowed, current_factors, depth):
        if time.time() > _DEADLINE:
            return
        
        if found_better[0]:
            return
        
        # If remaining is zero, we've found a valid decomposition
        if all(x == 0 for x in rem_sig):
            if len(current_factors) >= 1:
                found_better[0] = True
                better_factors[0] = list(current_factors)
            return
        
        if depth >= 12:
            return
        
        # Calculate remaining product
        rem_prod = 1
        for i, p in enumerate([2, 3, 5, 7, 11]):
            rem_prod *= p ** rem_sig[i]
        
        if rem_prod <= 1:
            return
        
        # Next factor must be > min_factor, <= max_allowed, and divide rem_prod
        # Also must be > 12
        start = max(min_factor + 1, 13)
        end = min(max_allowed, rem_prod)
        
        if start > end:
            return
        
        # Find divisors in [start, end]
        # Use binary search on sorted divisors
        import bisect
        lo = bisect.bisect_left(divisors, start)
        hi = bisect.bisect_right(divisors, end)
        
        for i in range(lo, hi):
            d = divisors[i]
            if rem_prod % d != 0:
                continue
            
            sig = div_sig[d]
            if not can_subtract(rem_sig, sig):
                continue
            
            new_rem = subtract_sig(rem_sig, sig)
            current_factors.append(d)
            
            if all(x == 0 for x in new_rem):
                # This is the last factor
                if len(current_factors) >= 1:
                    found_better[0] = True
                    better_factors[0] = list(current_factors)
                    current_factors.pop()
                    return
            else:
                # Continue searching
                # The remaining product must be achievable with factors > d
                dfs(new_rem, d, max_allowed, current_factors, depth + 1)
            
            current_factors.pop()
            if found_better[0]:
                return
    
    # Try to find a solution with max < 64
    # Search with max_allowed = 63
    dfs(target_sig, 12, 63, [], 0)
    
    if found_better[0] and better_factors[0] is not None:
        # Verify
        prod = 1
        for f in better_factors[0]:
            prod *= f
        if prod == target and all(f > 12 for f in better_factors[0]) and len(set(better_factors[0])) == len(better_factors[0]):
            return better_factors[0]
    
    # Try max = 62, 61, ... down to 13
    for max_allowed in range(63, 12, -1):
        if time.time() > _DEADLINE:
            break
        found_better[0] = False
        better_factors[0] = None
        dfs(target_sig, 12, max_allowed, [], 0)
        if found_better[0] and better_factors[0] is not None:
            prod = 1
            for f in better_factors[0]:
                prod *= f
            if prod == target and all(f > 12 for f in better_factors[0]) and len(set(better_factors[0])) == len(better_factors[0]):
                return better_factors[0]
    
    # Return the known best
    return best_factors
