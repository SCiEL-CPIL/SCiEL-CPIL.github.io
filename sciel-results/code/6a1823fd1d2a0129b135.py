import time
from math import factorial

_START = time.time()
_DEADLINE = _START + 85.0

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 85.0
    
    target = factorial(12)
    
    # Generate divisors of target > 12
    divisors = []
    for a in range(11):
        for b in range(6):
            for c in range(3):
                for d in range(2):
                    for e in range(2):
                        val = (2**a) * (3**b) * (5**c) * (7**d) * (11**e)
                        if val > 12:
                            divisors.append(val)
    
    divisors.sort()
    
    # Precompute divisors for each divisor
    # rem_divs[r] contains all d in divisors such that d < r and r % d == 0
    rem_divs = {}
    for r in divisors:
        rem_divs[r] = [d for d in divisors if d < r and r % d == 0]
        
    memo = {}
    choice = {}
    INF = float('inf')
    
    def dfs(rem, last):
        if time.time() > _DEADLINE:
            return INF
            
        if rem == 1:
            return 0
        
        # Pruning with memo
        # If we have seen this 'rem' with a 'last' <= current 'last', we can prune
        # because a smaller 'last' allows a superset of valid continuations
        if rem in memo:
            if last >= memo[rem]:
                return INF
            else:
                memo[rem] = last
        else:
            memo[rem] = last
            
        best_val = INF
        best_d = None
        
        # Option 1: Take rem as the last factor
        # Valid if rem > last (distinct/increasing) and rem > 12 (constraint)
        # rem is always > 12 because it's in divisors
        if rem > last:
            best_val = rem
            best_d = rem # Sentinel
            
        # Option 2: Split rem into d and rem/d
        # d must be > last and d < rem
        for d in rem_divs[rem]:
            if d <= last:
                continue
            # d < rem is guaranteed by rem_divs construction
            
            sub_val = dfs(rem // d, d)
            
            if sub_val < INF:
                cand = max(d, sub_val)
                if cand < best_val:
                    best_val = cand
                    best_d = d
                    
        choice[(rem, last)] = best_d
        return best_val

    dfs(target, 12)
    
    # Reconstruct factors
    rem = target
    last = 12
    factors = []
    
    # Safety check: if dfs didn't find a solution (should not happen)
    if (target, 12) not in choice:
        return [target]

    while rem > 1:
        d = choice.get((rem, last))
        if d is None:
            # Fallback
            break
        if d == rem:
            factors.append(d)
            rem = 1
        else:
            factors.append(d)
            last = d
            rem //= d
            
    if not factors:
        return [target]
        
    return factors
