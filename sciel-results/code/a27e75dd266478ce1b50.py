"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s kill

    # We want the minimum number k of distinct denominators d_i in [5, 10^7]
    # such that sum(1/d_i) = 1.
    #
    # Strategy: Use a bounded DFS with strong pruning. Search k from small to large.
    # Use integer arithmetic where possible for speed.
    
    best_result = None
    best_count = 257  # we need at most 256
    
    def check_time():
        return time.time() > deadline
    
    def dfs(remaining_num, remaining_den, min_den, current, depth, max_terms):
        """
        remaining_num / remaining_den is the remaining sum to achieve.
        min_den is the minimum allowed denominator (must be >= this).
        current is the list of denominators chosen so far.
        depth is the current depth (len(current)).
        max_terms is the maximum number of terms we're allowed to use total.
        """
        nonlocal best_result, best_count
        
        if check_time():
            return False
        
        # If we've found a solution, record it
        if remaining_num == 0:
            if len(current) < best_count:
                best_count = len(current)
                best_result = list(current)
            return True
        
        if remaining_num <= 0:
            return False
        
        # We need at least 1 more term
        if depth + 1 > max_terms:
            return False
        
        # Pruning: even with the smallest possible denominators, can we reach the target?
        # The maximum sum we can add with (max_terms - depth) terms each with denom >= min_den
        # is (max_terms - depth) / min_den
        # If remaining > that, prune.
        if remaining_num * min_den > remaining_den * (max_terms - depth):
            return False
        
        # Lower bound for next denominator d:
        # 1/d <= remaining => d >= remaining_den / remaining_num
        # Also d >= min_den
        d_low = max(min_den, (remaining_den + remaining_num - 1) // remaining_num)
        
        # Upper bound for d: 
        # If d is too large, 1/d is too small and we can't make up the difference
        # with the remaining terms. But a simple bound: d <= remaining_den * 2 + 1000
        # Actually, let's be more careful. If we use d, the new remaining is
        # (remaining_num*d - remaining_den) / (remaining_den*d)
        # This must be non-negative, so d >= remaining_den/remaining_num (already handled)
        # For upper bound, if d is very large, the remaining sum is almost unchanged,
        # but we're using up a term. Let's cap d reasonably.
        # A better upper bound: we need (max_terms - depth - 1) more terms, each with denom >= d+1
        # The minimum sum we can add is (max_terms - depth - 1) / (some large number)
        # But actually, the issue is that if d is too large, 1/d is tiny, and we need
        # the remaining sum to be achievable with fewer terms.
        # Let's just cap d at a reasonable value and rely on the recursion.
        d_high = min(10000000, remaining_den * 3 + 1000)
        
        # But also, if d_high < d_low, no solution
        if d_high < d_low:
            return False
        
        for d in range(d_low, d_high + 1):
            if check_time():
                return False
            
            # Check if 1/d <= remaining
            # 1/d <= remaining_num/remaining_den => remaining_den <= d * remaining_num
            if d * remaining_num < remaining_den:
                continue  # 1/d > remaining, skip (shouldn't happen with our d_low)
            
            # Actually, 1/d <= remaining means d >= remaining_den/remaining_num, which is our d_low
            # So for d >= d_low, 1/d <= remaining always holds.
            
            new_num = remaining_num * d - remaining_den
            new_den = remaining_den * d
            
            if new_num < 0:
                break  # 1/d > remaining, and for larger d, 1/d gets smaller, so this won't happen
            # Wait, that's wrong. As d increases, 1/d decreases, so new_num = remaining_num*d - remaining_den
            # increases. So if new_num < 0, we should continue, not break.
            # But d >= d_low = ceil(remaining_den/remaining_num), so remaining_num*d >= remaining_den,
            # so new_num >= 0. So this case shouldn't occur.
            
            if new_num == 0:
                # Found a solution
                current.append(d)
                if len(current) < best_count:
                    best_count = len(current)
                    best_result = list(current)
                current.pop()
                return True
            
            # Recurse
            current.append(d)
            dfs(new_num, new_den, d + 1, current, depth + 1, max_terms)
            current.pop()
            
            # If we found a better solution, we can stop
            if best_count <= depth + 1:
                return True
        
        return False
    
    # Try increasing k from 5 upward
    for k in range(5, 257):
        if check_time():
            break
        dfs(1, 1, 5, [], 0, k)
        if best_result is not None and best_count == k:
            # Found a solution with k terms, which is minimal since we're going in order
            return best_result
        if best_result is not None:
            # Found a solution with fewer than k terms (shouldn't happen since we go in order)
            return best_result
    
    if best_result is not None:
        return best_result
    
    # Fallback: use the original DP approach
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
            return reached[target]
    
    raise RuntimeError("no representation found")

# EVOLVE-BLOCK-END
