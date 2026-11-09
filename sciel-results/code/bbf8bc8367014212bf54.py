import time
from math import factorial

_START = time.time()
_DEADLINE = _START + 85

def construct():
    target = factorial(12)  # 479001600
    
    # Generate all divisors of target
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
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
    n_divs = len(valid_divs)
    
    best_max = [target]  # initial best: single factor
    best_factors = [[target]]
    
    def dfs(idx, prod, max_so_far, count):
        if time.time() > _DEADLINE:
            return
        if prod == target:
            if count >= 1:
                if max_so_far < best_max[0]:
                    best_max[0] = max_so_far
                    # We need to store the actual factors, but for the score only max matters
                    # We'll reconstruct at the end or just return max
            return
        if prod > target:
            return
        if count >= 12:
            return
        
        # Pruning: if max_so_far >= best_max[0], we can still continue 
        # but won't improve unless we find a solution with smaller max
        # Actually, if max_so_far is already >= best_max[0], any extension will have max >= max_so_far >= best_max[0]
        # So we can prune if we're looking for improvement
        # But we need at least one solution, so we only prune if we already have a better one
        if max_so_far >= best_max[0] and count > 0:
            # Even if we add more factors, max won't decrease
            # But we might still find a valid solution with same max - not useful
            # Only prune if we're confident no better solution exists from here
            # Actually, max_so_far is the current max, and we're adding larger factors (since sorted)
            # So new max will be >= max_so_far. If max_so_far >= best_max[0], no improvement possible.
            # But we still need to find SOME valid solution. If we already have one, we can prune.
            return
        
        remaining = target // prod
        if remaining <= 1:
            return
        
        # The next factor must be > valid_divs[idx-1] if idx > 0, and must divide remaining
        # Also, next factor must be > 12 (already ensured by valid_divs)
        
        for i in range(idx, n_divs):
            d = valid_divs[i]
            if d > remaining:
                break
            if remaining % d != 0:
                continue
            new_max = max(max_so_far, d)
            if new_max >= best_max[0] and count > 0:
                # Can't improve, and since factors are increasing, all subsequent will be >= d
                # So max will only increase
                break
            dfs(i + 1, prod * d, new_max, count + 1)
    
    dfs(0, 1, 0, 0)
    
    # Reconstruct: we need to actually return the factors
    # Let's redo with factor tracking
    # Actually, let's just return the best_max as a single-element list? No, we need the actual factors.
    # Let me redo this properly.
    
    # Reset and do it again with proper tracking
    best_max[0] = target
    best_factors[0] = [target]
    
    def dfs2(idx, prod, max_so_far, count, current_factors):
        if time.time() > _DEADLINE:
            return
        if prod == target:
            if count >= 1:
                if max_so_far < best_max[0]:
                    best_max[0] = max_so_far
                    best_factors[0] = list(current_factors)
            return
        if prod > target:
            return
        if count >= 12:
            return
        if max_so_far >= best_max[0] and count > 0:
            return
        
        remaining = target // prod
        if remaining <= 1:
            return
        
        for i in range(idx, n_divs):
            d = valid_divs[i]
            if d > remaining:
                break
            if remaining % d != 0:
                continue
            new_max = max(max_so_far, d)
            if new_max >= best_max[0] and count > 0:
                break
            current_factors.append(d)
            dfs2(i + 1, prod * d, new_max, count + 1, current_factors)
            current_factors.pop()
    
    dfs2(0, 1, 0, 0, [])
    
    return best_factors[0]
