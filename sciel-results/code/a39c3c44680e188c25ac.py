"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    # Known minimal solutions search: try k=4,5,6 with DFS
    # Use integer arithmetic via Fraction
    
    def dfs(terms_left, min_d, remaining, path):
        if time.time() > deadline:
            return None
        if terms_left == 0:
            if remaining == 0:
                return list(path)
            return None
        if remaining <= 0:
            return None
        
        # Prune: max possible sum with terms_left terms starting at min_d
        # 1/min_d + 1/(min_d+1) + ... < terms_left / min_d
        if Fraction(terms_left, min_d) < remaining:
            return None
        
        # d must satisfy 1/d <= remaining => d >= 1/remaining
        # and d >= min_d
        # Also 1/d + 1/(d+1) + ... >= remaining for feasibility
        # Upper bound on d: 1/d >= remaining / terms_left => d <= terms_left / remaining
        max_d = int(terms_left / remaining) if remaining > 0 else 10000000
        max_d = min(max_d + 1, 10000000)
        min_d_actual = max(min_d, 5)
        
        for d in range(min_d_actual, max_d + 1):
            if time.time() > deadline:
                return None
            frac = Fraction(1, d)
            if frac > remaining:
                break
            new_rem = remaining - frac
            path.append(d)
            result = dfs(terms_left - 1, d + 1, new_rem, path)
            path.pop()
            if result is not None:
                return result
        return None
    
    for k in range(4, 10):
        if time.time() > deadline:
            break
        result = dfs(k, 5, Fraction(1, 1), [])
        if result is not None:
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result
    
    # Fallback
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            result = reached[target]
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
