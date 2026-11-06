"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _get_divisors(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    divs.sort(reverse=True)
    return divs

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin before 60s limit
    
    n = 5040
    target = 4979
    divisors = _get_divisors(n)  # 60 divisors, sorted descending
    
    # Quick check: can we hit target with few divisors?
    # Try k = 1, 2, 3, ... up to some reasonable limit
    max_k = 20  # upper bound on number of divisors to try
    
    for k in range(1, max_k + 1):
        if time.time() > deadline:
            break
        # For each combination of k divisors, check if sum == target
        # Pruning: if partial sum > target, skip; if remaining max possible < needed, skip
        found = None
        
        def dfs(start_idx, remaining, chosen, depth):
            if time.time() > deadline:
                return None
            if depth == k:
                if remaining == 0:
                    return list(chosen)
                return None
            if remaining < 0:
                return None
            
            # Pruning: even if we take all remaining largest divisors, can we reach 'remaining'?
            # Max possible sum from remaining divisors starting at start_idx
            max_possible = sum(divisors[start_idx:start_idx + (k - depth)])
            if max_possible < remaining:
                return None
            
            # Min possible: take smallest available
            # Actually we need exactly 'remaining' from the rest
            for i in range(start_idx, len(divisors)):
                d = divisors[i]
                if d > remaining:
                    continue
                chosen.append(d)
                result = dfs(i + 1, remaining - d, chosen, depth + 1)
                if result is not None:
                    return result
                chosen.pop()
            return None
        
        result = dfs(0, target, [], 0)
        if result is not None:
            # Verify
            assert len(result) <= 256
            assert all(d in set(divisors) for d in result)
            assert len(set(result)) == len(result)  # distinct
            assert sum(result) == target
            return result
    
    # Fallback: greedy (should not reach here for this instance)
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    assert remainder == 0
    return chosen

# EVOLVE-BLOCK-END
