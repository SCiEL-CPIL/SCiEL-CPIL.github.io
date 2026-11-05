import time
from math import factorial

_TARGET = factorial(12)  # 479001600
_MIN_FACTOR = 13
_MAX_COUNT = 12

def _get_divisors(n):
    """Get all divisors of n."""
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    return divs

# Precompute divisors of 12! that are > 12
_ALL_DIVS = [d for d in _get_divisors(_TARGET) if d > _MIN_FACTOR]
_ALL_DIVS.sort()
_DIV_SET = set(_ALL_DIVS)

# Index divisors for faster set operations if needed, but set of ints is fine for <800 items.

def _construct_heuristic(timeout=1.0):
    """
    Quick heuristic to find a valid solution with reasonably small max factor.
    Uses randomized greedy/backtracking.
    """
    start_time = time.time()
    best_factors = None
    best_max = float('inf')
    
    # We will try to find a solution with a specific number of factors, or just let it flow.
    # A good strategy is to pick divisors close to the geometric mean.
    
    def dfs(remaining, used, cur, depth):
        nonlocal best_factors, best_max
        if time.time() - start_time > timeout:
            return False
        if remaining == 1:
            if cur:
                mx = max(cur)
                if mx < best_max:
                    best_max = mx
                    best_factors = list(cur)
            return True
        if len(cur) >= _MAX_COUNT:
            return False
            
        # Lower bound on max factor:
        # If we have k slots left, the max must be at least remaining ** (1/k).
        slots_left = _MAX_COUNT - len(cur)
        if slots_left == 0:
            return False
        lb = remaining ** (1.0 / slots_left)
        if lb >= best_max:
            return False
            
        # Candidate factors: divisors of remaining, not used, < best_max
        cands = []
        for d in _ALL_DIVS:
            if d > remaining:
                break
            if d >= best_max:
                continue
            if remaining % d == 0 and d not in used:
                cands.append(d)
        
        if not cands:
            return False
            
        # Sort candidates by closeness to geometric mean of remaining for remaining slots
        gm = remaining ** (1.0 / slots_left)
        cands.sort(key=lambda x: abs(x - gm))
        
        for d in cands:
            used.add(d)
            cur.append(d)
            dfs(remaining // d, used, cur, depth + 1)
            cur.pop()
            used.remove(d)
            
            if best_factors is not None:
                # If we found a solution, we can tighten the bound and maybe restart or continue
                pass
                
        return best_factors is not None

    dfs(_TARGET, set(), [], 0)
    return best_factors

def construct():
    start = time.time()
    deadline = start + 55.0 # Safety margin for 60s limit
    
    # 1. Get an initial solution with the heuristic
    initial = _construct_heuristic(timeout=2.0)
    if initial:
        best_max = max(initial)
        best_factors = initial
    else:
        # Fallback if heuristic fails completely (shouldn't happen)
        # Just return a large valid decomposition
        # 12! = 1*2*... *12, but factors must be > 12.
        # We can combine 1,2,3... into factors > 12.
        # e.g. 13 is not a divisor. 
        # Let's just use a known valid one if we find one quickly, or rely on DFS to find one from inf.
        # But DFS from inf might be slow. 
        # Let's force a quick search with a large bound.
        best_max = _TARGET
        best_factors = None

    # 2. Run full Branch and Bound
    # State: remaining, used (frozenset for hashability in recursion if we used memo, but we don't memoize state, just path), cur
    # To speed up, we don't use memoization on (remaining, used) because 'used' is large.
    # We rely on pruning.
    
    def dfs(remaining, used, cur, max_so_far):
        nonlocal best_max, best_factors
        
        if time.time() > deadline:
            return
            
        if max_so_far >= best_max:
            return
            
        if remaining == 1:
            if cur and len(cur) <= _MAX_COUNT:
                # Valid solution
                if max_so_far < best_max:
                    best_max = max_so_far
                    best_factors = list(cur)
            return
            
        if len(cur) >= _MAX_COUNT:
            return
            
        slots_left = _MAX_COUNT - len(cur)
        
        # Pruning 1: Geometric mean lower bound
        lb = remaining ** (1.0 / slots_left)
        if lb >= best_max:
            return
            
        # Pruning 2: Min product of distinct factors > 12
        # The smallest possible product for 'slots_left' distinct factors > 12 
        # is 13 * 14 * ... * (12 + slots_left).
        # If remaining < this product, we can't fill the slots with distinct factors > 12.
        # However, we don't HAVE to use exactly slots_left factors. We can use fewer.
        # If we use k factors, the product is remaining.
        # The minimal factor value is > 12.
        # So max factor must be at least remaining / (product of largest k-1 distinct factors < best_max).
        # This is complex. The geometric mean bound is usually strong enough.
        
        # Generate candidates
        # Divisors of remaining, > 12, not used, < best_max
        cands = []
        for d in _ALL_DIVS:
            if d > remaining:
                break
            if d >= best_max:
                continue
            if remaining % d == 0 and d not in used:
                cands.append(d)
                
        if not cands:
            return
            
        # Ordering:
        # To minimize the max, we want to avoid large factors.
        # However, we also need to be able to factor the rest.
        # Sorting by value (ascending) ensures we try small factors first.
        # But sometimes a slightly larger factor is needed to allow the rest to be small.
        # A good heuristic: sort by d, but prioritize those that leave a 'balanced' remaining product.
        # Let's try sorting by d (ascending). This usually finds solutions with small max faster 
        # because we prune the "large factor" branches later? 
        # Actually, if we pick a small d, the remaining product is large, so the next factors might need to be large.
        # If we pick a d close to GM, the remaining factors are more balanced.
        
        # Let's sort by closeness to GM, but ensure we explore balanced options first.
        gm = remaining ** (1.0 / slots_left)
        cands.sort(key=lambda d: (abs(d - gm), d))
        
        for d in cands:
            new_max = max(max_so_far, d)
            if new_max >= best_max:
                continue
            used.add(d)
            cur.append(d)
            dfs(remaining // d, used, cur, new_max)
            cur.pop()
            used.remove(d)
            
            if time.time() > deadline:
                return

    if best_factors is None:
        # If no initial solution, we need to find one.
        # The DFS above starts with best_max = _TARGET, so it will search.
        dfs(_TARGET, set(), [], 0)
    else:
        dfs(_TARGET, set(best_factors), list(best_factors), best_max)
        
    if best_factors:
        # Final validation
        if len(best_factors) <= _MAX_COUNT and all(f > 12 for f in best_factors) and len(set(best_factors)) == len(best_factors):
            prod = 1
            for f in best_factors:
                prod *= f
            if prod == _TARGET:
                return best_factors
                
    # Fallback: return a valid decomposition even if not optimal
    # We can construct one by combining primes
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    # We need distinct factors > 12.
    # Example:
    # 13 is not a divisor.
    # 14 = 2*7
    # 15 = 3*5
    # 16 = 2^4
    # 18 = 2*3^2
    # 20 = 2^2*5
    # 21 = 3*7 -- 7 used
    # Let's just return the best found. If none, return a large number (which is a valid decomposition of 1 factor)
    # But 1 factor is 12! which is > 12.
    # Is 12! distinct? Yes.
    # Is it > 12? Yes.
    # So [12!] is always a valid fallback.
    return [_TARGET]
