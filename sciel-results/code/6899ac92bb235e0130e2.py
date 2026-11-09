import time
from math import factorial

_START = time.time()
_DEADLINE = _START + 85  # safety margin before 90s kill

def construct():
    global _START, _DEADLINE
    target = factorial(12)  # 479001600
    
    # Generate all divisors of target
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # Enumerate all divisors
    divisors = []
    for a in range(11):  # 2^a
        for b in range(6):  # 3^b
            for c in range(3):  # 5^c
                for d in range(2):  # 7^d
                    for e in range(2):  # 11^e
                        d_val = (2**a) * (3**b) * (5**c) * (7**d) * (11**e)
                        divisors.append(d_val)
    
    # Filter: must be > 12
    valid_divs = [d for d in divisors if d > 12]
    valid_divs.sort()
    valid_divs_set = set(valid_divs)
    
    best_max = target  # single factor
    best_factors = [target]
    
    # Heuristic: Try to find a good initial solution to tighten best_max
    # Greedy: pick smallest divisor > 12 that divides remaining
    rem = target
    greedy_factors = []
    min_d = 13
    while rem > 1:
        if time.time() > _DEADLINE:
            break
        found = False
        for d in valid_divs:
            if d < min_d:
                continue
            if d > rem:
                break
            if rem % d == 0:
                greedy_factors.append(d)
                rem //= d
                min_d = d + 1
                found = True
                break
        if not found:
            # If no divisor found, rem must be 1 or a prime > 12 (but rem divides target, so it's composite or 1)
            # If rem > 1 and no divisor found, it means rem itself must be the last factor
            # Check if rem is a valid divisor
            if rem > 12 and rem >= min_d:
                greedy_factors.append(rem)
                rem = 1
            else:
                break # Failed
    
    if rem == 1 and len(greedy_factors) <= 12:
        g_max = max(greedy_factors)
        if g_max < best_max:
            best_max = g_max
            best_factors = greedy_factors
    
    def dfs(remaining, current_factors, min_allowed, max_so_far, depth):
        nonlocal best_max, best_factors
        
        if time.time() > _DEADLINE:
            return
        
        if remaining == 1:
            # Found a complete factorization
            if len(current_factors) >= 1:
                m = max_so_far
                if m < best_max:
                    best_max = m
                    best_factors = list(current_factors)
            return
        
        # Optimization: If remaining is small enough that it cannot be split into two factors > 12
        # Smallest product of two distinct integers > 12 is 13 * 14 = 182.
        # If remaining < 182, it must be a single factor.
        if remaining < 182:
            if remaining > 12 and remaining >= min_allowed:
                new_max = max(max_so_far, remaining)
                if new_max < best_max:
                    best_max = new_max
                    best_factors = current_factors + [remaining]
            return
        
        # Also check if remaining itself is a valid factor (and update best_max)
        # This handles the case where we might prefer remaining as a single factor over splitting it
        # if splitting leads to a larger max (unlikely, but possible if one factor is huge? No, factors are < remaining)
        # Actually, if remaining is a divisor, taking it as one factor gives max = max(max_so_far, remaining).
        # Splitting it into a, b gives max = max(max_so_far, a, b). Since a,b < remaining, this is usually better.
        # So we only need to check this if we CAN'T split it further?
        # We already handled remaining < 182.
        # If remaining >= 182, we should try to split it.
        
        # Iterate through divisors
        for d in valid_divs:
            if time.time() > _DEADLINE:
                return
            
            if d < min_allowed:
                continue
            if d > remaining:
                break
            if d > best_max:
                break  # since sorted, no point continuing
            
            if remaining % d != 0:
                continue
            
            new_remaining = remaining // d
            new_max = max(max_so_far, d)
            
            if new_max >= best_max:
                continue
            
            # Pruning: if new_remaining < 13, it's invalid (unless it's 1)
            # But we handled remaining==1 at start.
            # If new_remaining > 1, it must be >= 13 (since factors > 12)
            # Actually, if new_remaining is a prime < 13, it's invalid.
            # But new_remaining divides target.
            
            current_factors.append(d)
            dfs(new_remaining, current_factors, d + 1, new_max, depth + 1)
            current_factors.pop()
    
    dfs(target, [], 13, 0, 0)
    
    if time.time() > _DEADLINE - 5:
        # Return best found so far
        return best_factors
    
    return best_factors
