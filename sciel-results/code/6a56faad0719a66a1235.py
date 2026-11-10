"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    target = Fraction(5, 31)
    deadline = time.time() + 85.0  # 85s safety margin out of 90s
    best_solution = None
    best_max_den = float('inf')
    MAX_TERMS = 256
    MAX_DEN_LIMIT = 10000000
    
    # Precompute a table of max sum of reciprocals of N distinct integers starting from min_d
    # This is used for pruning: if sum_{i=0}^{N-1} 1/(min_d+i) < remaining, we can prune.
    # We approximate this with the integral or a simple bound.
    # Sum_{i=0}^{N-1} 1/(min_d+i) < 1/(min_d) + sum_{i=1}^{N-1} 1/(min_d) = N/min_d
    # Actually, a tighter bound: Sum < 1/min_d + Integral_{min_d}^{min_d+N-1} (1/x) dx
    # But N/min_d is fast and safe (overestimates the sum, so if even this max possible sum < remaining, prune).
    
    def max_possible_sum(min_d, count):
        """Upper bound on sum of 'count' distinct reciprocals >= min_d."""
        if count <= 0:
            return 0
        # Sum_{i=0}^{count-1} 1/(min_d + i)
        # For large min_d, approx count/min_d.
        # For small min_d, we can compute or use a tighter bound.
        # Let's use the integral bound: 1/min_d + ln((min_d+count-1)/min_d)
        # This is an upper bound because 1/x is convex and decreasing.
        # Actually, sum < 1/min_d + integral_{min_d}^{min_d+count-1} 1/x dx
        # = 1/min_d + ln((min_d+count-1)/min_d)
        if min_d == 0:
            return float('inf')
        return 1.0/min_d + (min_d + count - 1) * 1.0 / min_d # Simple N/min_d bound is safer and faster
        
        # Let's stick to a very safe, fast bound: count / min_d
        # If remaining > count / min_d, it's impossible.
        
    def dfs(remaining, min_d, terms_left, used, current):
        nonlocal best_solution, best_max_den
        
        if time.time() > deadline:
            return False
        
        if remaining == 0:
            if current:
                max_den = max(current)
                if max_den < best_max_den:
                    best_max_den = max_den
                    best_solution = current[:]
                return True
            return False
        
        if terms_left == 0:
            return False
            
        # Pruning 1: Even with the smallest possible denominators, we can't reach the target
        # Max sum we can achieve with 'terms_left' terms starting from 'min_d'
        # Approx max sum = sum_{i=0}^{terms_left-1} 1/(min_d+i)
        # We use the bound: terms_left / min_d
        # If remaining > terms_left / min_d, prune.
        if remaining > Fraction(terms_left, min_d):
            return False
            
        # Pruning 2: We must find a denominator d >= min_d such that 1/d <= remaining
        # So d >= 1/remaining
        min_d_required = max(min_d, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        
        # Pruning 3: If we use the largest possible denominator (limited by best_max_den), 
        # can we still represent the remainder?
        # This is complex, so we rely on the search.
        
        # Branch and Bound: Iterate d from min_d_required upwards
        # We limit the search depth of d to keep it fast.
        # If d is too large, 1/d becomes very small, and we might not be able to reach 'remaining' 
        # with the remaining terms if d is too big? No, large d means small term, so we need MORE terms.
        # But we have a fixed 'terms_left'.
        # Actually, if d is very large, the remaining fraction changes slightly.
        
        # Heuristic: We don't want to try infinite d.
        # If 1/d < remaining / terms_left, then even if we pick the smallest possible d, 
        # the average term size is too small?
        # No, we want to find ANY solution.
        
        # Let's bound d. 
        # If d > best_max_den, we can't improve the score (since we minimize max den).
        # But we might find a solution with a smaller max den later? No, if d > best_max_den, 
        # the current solution's max den will be at least d, which is worse.
        # So we can prune d > best_max_den.
        
        max_d_to_try = best_max_den - 1 if best_max_den < float('inf') else 10000000
        
        # Optimization: If min_d_required > max_d_to_try, no solution better than current.
        if min_d_required > max_d_to_try:
            return False
            
        # To speed up, we can skip some d values if we are confident.
        # But for correctness, we must check.
        
        # Limit the number of d's we try in one level to avoid explosion.
        # If terms_left is large, we can be more aggressive?
        
        # Let's try a range of d's.
        # If remaining is small, d must be large.
        # If remaining is large, d must be small.
        
        # We will iterate d from min_d_required to max_d_to_try.
        # But this range could be huge.
        # We can use a step or limit.
        # However, for Egyptian fractions, the optimal solutions often have denominators close to the greedy choice.
        
        # Let's try a window around the greedy choice.
        # Greedy d_g = ceil(1/remaining)
        d_greedy = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
        if d_greedy < min_d_required:
            d_greedy = min_d_required
            
        # Try d in a range [d_start, d_end]
        # d_start = min_d_required
        # d_end = min(max_d_to_try, d_greedy + 100) # Heuristic: don't go too far from greedy
        
        # Actually, if we go far from greedy, the remainder changes significantly.
        # Let's try a wider window if we have time.
        
        d_start = min_d_required
        d_end = min(max_d_to_try, d_greedy + 50)
        
        # If the window is empty, return
        if d_start > d_end:
            return False
            
        for d in range(d_start, d_end + 1):
            if time.time() > deadline:
                return False
                
            if d in used:
                continue
                
            # Check if 1/d <= remaining
            if Fraction(1, d) > remaining:
                continue
                
            new_remaining = remaining - Fraction(1, d)
            
            # Recurse
            used.add(d)
            current.append(d)
            
            # Pruning in recursion:
            # If new_remaining > 0, we need more terms.
            # If new_remaining == 0, we found a solution.
            
            if dfs(new_remaining, d + 1, terms_left - 1, used, current):
                # Found a better solution? The dfs updates best_solution.
                # We can return True to stop this branch if we found a solution 
                # that satisfies the bound? 
                # Actually, we want to find the MINIMUM max den.
                # If we found a solution, its max den is <= d (if d is the max) or larger.
                # But we are iterating d upwards. So smaller d's are tried first.
                # If we find a solution, it might not be the global optimum for max den 
                # if a later branch with larger d but smaller *other* denominators exists?
                # No, max den is the max of all. If we fix d, the max is at least d.
                # So if we find a solution with max_den = M, any solution with d > M is worse.
                # Since we iterate d from small to large, the first solution we find 
                # in a branch doesn't necessarily have the smallest max_den globally, 
                # but it gives an upper bound.
                
                # We continue searching to see if we can find a solution with a smaller max_den.
                # But if we found a solution, best_max_den is updated.
                # The loop will prune d > best_max_den in subsequent iterations.
                
                pass # Continue to find potentially better solutions
            
            current.pop()
            used.remove(d)
            
        return False

    # Initial call
    # We need to start with some denominators.
    # The first denominator d1 must be >= 7.
    # We can start the DFS from the beginning.
    # To make it faster, we can pre-select the first few denominators using a heuristic 
    # and then DFS on the rest.
    
    # Let's try a hybrid: 
    # 1. Get a good initial solution with Greedy.
    # 2. Use DFS to improve it by exploring alternatives for the first K denominators.
    
    def greedy_expand(frac, min_d=1, used=None):
        terms = []
        r = frac
        min_d = min_d
        used = used or set()
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d < min_d:
                d = min_d
            while d in used:
                d += 1
            terms.append(d)
            r -= Fraction(1, d)
            min_d = d + 1
            used.add(d)
        return terms

    baseline = greedy_expand(target)
    if baseline:
        best_solution = baseline[:]
        best_max_den = max(baseline)
        
    # Now, use DFS to improve.
    # We will fix the first 2 denominators and DFS on the rest.
    # Or just DFS from the start with a depth limit?
    # DFS from start might be too slow if the branching factor is high.
    # Let's fix the first denominator d1 in a small range, and DFS for the rest.
    
    # The first denominator d1 is at least 7.
    # Let's try d1 from 7 to 15.
    for d1 in range(7, 16):
        if time.time() > deadline:
            break
        if d1 > best_max_den:
            break
            
        remaining1 = target - Fraction(1, d1)
        if remaining1 <= 0:
            continue
            
        # DFS for the rest
        # We have MAX_TERMS - 1 terms left.
        # Min d for next is d1 + 1.
        
        # To speed up DFS, we can use a local deadline or just rely on the global one.
        # The DFS function uses the global deadline.
        
        # We need to pass a fresh used set and current list.
        used = {d1}
        current = [d1]
        
        # We want to find a solution with max_den < best_max_den.
        # So in DFS, we can prune if d >= best_max_den.
        
        # Let's call DFS
        dfs(remaining1, d1 + 1, MAX_TERMS - 1, used, current)
        
    # If no better solution found, return the baseline
    if best_solution is None:
        return baseline
        
    return best_solution
# EVOLVE-BLOCK-END
