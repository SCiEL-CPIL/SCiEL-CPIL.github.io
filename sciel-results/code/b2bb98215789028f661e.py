import time
from math import factorial
from itertools import combinations

_START = time.time()
_DEADLINE = _START + 85  # safety margin before 90s kill

def construct():
    target = factorial(12)  # 479001600 = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    
    # Generate all divisors of target
    # 2^a * 3^b * 5^c * 7^d * 11^e where a in [0,10], b in [0,5], c in [0,2], d in [0,1], e in [0,1]
    divisors = set()
    for a in range(11):
        for b in range(6):
            for c in range(3):
                for d in range(2):
                    for e in range(2):
                        val = (2**a) * (3**b) * (5**c) * (7**d) * (11**e)
                        if val > 12:
                            divisors.add(val)
    
    valid_divs = sorted(divisors)
    valid_divs_set = set(valid_divs)
    
    best_max = target  # initial: single factor
    best_factors = [target]
    
    # DFS: pick distinct divisors from valid_divs such that product == target
    # Minimize the maximum factor
    # Pruning: 
    #   - if current max >= best_max, prune
    #   - if remaining product < smallest available unused divisor, prune
    #   - if we've used 12 factors, stop
    
    def dfs(remaining, used_set, current_factors, current_max, idx):
        nonlocal best_max, best_factors
        
        if time.time() > _DEADLINE:
            return
        
        if remaining == 1:
            if current_factors:
                m = max(current_factors)
                if m < best_max:
                    best_max = m
                    best_factors = list(current_factors)
            return
        
        if len(current_factors) >= 12:
            return
        
        # Prune: if remaining is not divisible by any available divisor > 12
        # Also, remaining itself must be achievable
        
        # Try adding one more factor
        for i in range(idx, len(valid_divs)):
            d = valid_divs[i]
            
            # Prune: if d > remaining, break (sorted)
            if d > remaining:
                break
            
            # Prune: if d doesn't divide remaining
            if remaining % d != 0:
                continue
            
            # Prune: if d is already used (shouldn't happen with distinct, but check)
            if d in used_set:
                continue
            
            # Prune: if d >= best_max and we already have factors, the max won't improve
            # Actually, even if d < best_max, we need the rest to also be < best_max
            # Better prune: if d >= best_max, then max will be at least d >= best_max, no improvement
            if d >= best_max:
                break  # since sorted, all subsequent are >= d >= best_max
            
            new_remaining = remaining // d
            new_max = max(current_max, d)
            
            # Prune: if new_max >= best_max, no improvement possible
            if new_max >= best_max:
                continue
            
            # Prune: new_remaining must be 1 or must have divisors > 12 that are distinct from used
            if new_remaining > 1:
                # Check if new_remaining can be formed from distinct divisors > 12 not in used_set
                # Quick check: new_remaining must be > 12 or 1
                # If new_remaining <= 12 and new_remaining != 1, it can't be a valid factor
                # But we could have multiple more factors
                # For now, just continue; deeper pruning in recursion
                pass
            
            used_set.add(d)
            current_factors.append(d)
            dfs(new_remaining, used_set, current_factors, new_max, i + 1)
            current_factors.pop()
            used_set.discard(d)
            
            if time.time() > _DEADLINE:
                return
    
    dfs(target, set(), [], 0, 0)
    
    if best_factors:
        return best_factors
    return [target]
