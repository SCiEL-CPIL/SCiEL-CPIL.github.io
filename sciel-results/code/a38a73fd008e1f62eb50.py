"""Combined approach: subset-sum for fast valid solution + DFS for minimization."""
import time
from fractions import Fraction
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 90s kill
    
    # Phase 1: Fast subset-sum over divisors of a common denominator
    # to get a guaranteed valid solution.
    target = 2520  # LCM(5..15)
    # Collect all valid denominators d in [5, target] where target % d == 0
    # weight = target // d
    items = []
    for d in range(5, target + 1):
        if target % d == 0:
            w = target // d
            items.append((d, w))
    
    # Subset-sum: find subset of weights summing to target
    # Use dict: sum -> (count, denominators)
    # We want to minimize count, so track best (min count) for each sum
    best = {0: (0, [])}  # sum -> (count, denominators)
    for d, w in items:
        # Process in reverse to avoid using same item twice
        # But we want to minimize count, so we track best count per sum
        new_entries = {}
        for s, (cnt, dens) in best.items():
            ns = s + w
            if ns <= target and ns not in best:
                new_entries[ns] = (cnt + 1, dens + [d])
            elif ns in best:
                if cnt + 1 < best[ns][0]:
                    new_entries[ns] = (cnt + 1, dens + [d])
        for s, val in new_entries.items():
            if s not in best or val[0] < best[s][0]:
                best[s] = val
    
    if target in best:
        fallback_solution = best[target][1]
        fallback_count = best[target][0]
    else:
        fallback_solution = None
        fallback_count = float('inf')
    
    # Phase 2: Try DFS to find a solution with fewer terms
    # Theoretical minimum: 7 terms max sum = 1/5+1/6+1/7+1/8+1/9+1/10+1/11 ≈ 0.9366 < 1
    # So minimum is at least 8.
    
    def dfs(remaining, start_denom, terms_left, current, best_count):
        if time.time() > deadline:
            return None
        if remaining == 0:
            return current[:]
        if terms_left == 0:
            return None
        if remaining < 0:
            return None
        if len(current) >= best_count:
            return None
        
        # Pruning: max possible sum with terms_left terms from start_denom
        # sum(1/i for i in range(start_denom, start_denom + terms_left))
        max_possible = Fraction(0, 1)
        for i in range(start_denom, start_denom + terms_left):
            max_possible += Fraction(1, i)
        if max_possible < remaining:
            return None
        
        if terms_left == 1:
            if remaining.numerator == 1:
                d = remaining.denominator
                if d >= start_denom and d >= 5 and d <= 10000000:
                    return current + [d]
            return None
        
        # Bounds on d
        # 1/d <= remaining => d >= ceil(1/remaining)
        min_d = max(start_denom, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        
        # 1/d + (terms_left-1)/(d+1) >= remaining (rough upper bound)
        # Simplified: d <= terms_left / remaining
        max_d = (terms_left * remaining.denominator) // remaining.numerator + 1
        max_d = min(max_d, 10000000)
        
        for d in range(min_d, max_d + 1):
            next_remaining = remaining - Fraction(1, d)
            if next_remaining < 0:
                break
            result = dfs(next_remaining, d + 1, terms_left - 1, current + [d], best_count)
            if result is not None:
                return result
        return None
    
    best_solution = fallback_solution
    best_count = fallback_count
    
    # Try increasing k from 8 up to best_count - 1
    for k in range(8, min(best_count, 257)):
        if time.time() > deadline:
            break
        result = dfs(Fraction(1, 1), 5, k, [], best_count)
        if result is not None:
            # Verify
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result):
                best_solution = result
                best_count = k
                break  # Found minimal, stop
    
    if best_solution is None:
        raise RuntimeError("no representation found")
    
    return best_solution
