import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin
    
    target = Fraction(5, 31)
    best_max = float('inf')
    best_solution = None
    
    # Helper to update best solution
    def update_best(sol):
        nonlocal best_max, best_solution
        if sol and len(sol) <= 256 and len(set(sol)) == len(sol):
            if all(1 <= x <= 10000000 for x in sol):
                m = max(sol)
                if m < best_max:
                    best_max = m
                    best_solution = sol
    
    # The core number-theoretic search
    # State: (num, den, used_set, path)
    # To find next denominator d for remainder num/den:
    # We need 1/d < num/den => d > den/num
    # Also 1/d - num/den = (den - num*d) / (den*d)
    # For the remainder to be representable with integers, it's often beneficial
    # to look for d such that the new remainder has small factors or is 0.
    # A strong heuristic for minimizing MAX denominator is to try small d first.
    
    # We will use a Depth First Search with pruning.
    # Pruning:
    # 1. d > den/num
    # 2. d not in used
    # 3. d <= current_best_max (if we are trying to beat it, though we don't know it yet)
    # 4. d <= 10,000,000
    # 5. If num/den is small, d cannot be too large? No, d can be large.
    # But if we fix a small d, the remainder might be small.
    
    # Let's refine the search.
    # We want to minimize max(d1, d2, ..., dk).
    # This is equivalent to finding a set of distinct denominators D such that
    # sum(1/d) = 5/31 and max(D) is minimized.
    
    # We can iterate over the potential maximum denominator M.
    # For a fixed M, does there exist a set of distinct denominators in [1, M]
    # summing to 5/31?
    # This is a subset sum problem with rationals, which is hard.
    
    # Alternative: DFS with branching on small denominators.
    # Since 5/31 ~ 0.16, the smallest denominator must be at least 6 (1/6 > 5/31, 1/7 < 5/31).
    # Actually 1/6 = 0.166 > 0.161, so 6 is too big? No, 1/6 > 5/31.
    # 5/31 = 0.16129...
    # 1/6 = 0.1666... > 5/31. So 1/6 is too large a term?
    # If we pick 6, remainder is 5/31 - 1/6 = (30-31)/186 = -1/186. Negative.
    # So we cannot pick 6.
    # Smallest possible denominator is 7. 1/7 = 0.1428... < 0.1612...
    # Remainder: 5/31 - 1/7 = (35-31)/217 = 4/217.
    
    # So d1 >= 7.
    
    def dfs(num, den, used, path, current_max):
        nonlocal best_max, best_solution, deadline
        
        if time.time() > deadline:
            return
        
        if num == 0:
            if path:
                update_best(path)
            return
        
        if num < 0:
            return
            
        # If we have too many terms
        if len(path) >= 255:
            return
            
        # Pruning: if the maximum possible sum of remaining terms (using smallest available denominators)
        # is less than num/den, we can prune.
        # But this is hard to compute exactly without knowing which denominators are used.
        # A simpler bound: if num/den > 1/(min_possible_d) + ...
        
        # Let's just try denominators d starting from ceil(den/num)
        min_d = den // num + 1
        if min_d in used:
            min_d += 1
            while min_d in used:
                min_d += 1
                
        # We want to minimize the MAX denominator.
        # If we pick a small d, the remainder is larger, potentially requiring larger denominators later?
        # Or picking a d close to den/num makes the remainder small, potentially allowing smaller denominators?
        # Actually, if we pick d close to den/num, the remainder is small, so subsequent denominators can be large.
        # If we pick d much larger than den/num, the remainder is close to num/den, so we need many terms.
        
        # To minimize the MAX, we should probably avoid huge denominators.
        # So we should try d in a reasonable range.
        
        # Upper bound for d in this step:
        # If d is too large, the remainder is approx num/den.
        # We need to complete num/den with remaining terms.
        # If we use k terms, each at least min_d, sum <= k * (1/min_d).
        # So k >= num/den * min_d.
        # If k > 255, it's impossible.
        
        max_terms_left = 255 - len(path)
        
        # If 1/min_d * max_terms_left < num/den, then even using the smallest possible denominators
        # for all remaining terms won't sum enough.
        # num/den > max_terms_left / min_d
        # num * min_d > den * max_terms_left
        if num * min_d > den * max_terms_left:
            return
            
        # Now iterate d
        # We should limit the range of d to avoid infinite loops or too slow searches.
        # If d is very large, the remainder doesn't change much.
        # Let's try d up to some limit, or until we find a solution.
        
        # Heuristic: Try d in a window around den/num.
        # Also, if we have a current best_max, we can prune d > best_max if we are trying to improve?
        # No, we want to find ANY solution with max < best_max.
        # So d must be < best_max if we are in a branch that already has max < best_max?
        # Actually, if any d in the path is > best_max, we can prune.
        
        # Let's iterate d from min_d upwards.
        # To speed up, we can skip d that don't lead to "nice" remainders.
        # A remainder num'/den' is "nice" if it has small factors.
        
        # For efficiency, let's limit the number of d we try per level.
        # Or use a smarter selection.
        
        # Let's try d from min_d to min_d + 10000 or so?
        # If min_d is large, we might need to go further.
        
        # Better: iterate d such that the new remainder num'/den' has num' < some threshold?
        
        # Let's just do a bounded search for d.
        # If we don't find a solution with d < best_max, we might need larger d.
        # But if best_max is inf, we can't bound it.
        
        # Let's use a heuristic: try d in [min_d, min_d + 1000] first.
        # If that fails, try [min_d + 1000, min_d + 2000], etc.
        # But this is slow.
        
        # Alternative: Randomized search with restarts?
        # No, let's stick to DFS but with strong pruning.
        
        # Pruning: If num/den is very small, we need large d.
        # If num/den is large, we need small d.
        
        # Let's try a specific range of d.
        # Since we want to minimize MAX, we prefer small d.
        # So we iterate d from min_d upwards.
        
        # To prevent explosion, we limit the depth or the number of branches.
        # But Egyptian fractions can have many terms.
        
        # Let's try to find a solution with few terms first, as that often implies smaller max denominators?
        # Not necessarily.
        
        # Let's just run DFS.
        
        # Optimization: If we have a best_max, we can stop trying d if d > best_max and current_max < best_max?
        # No, if d > best_max, then max(path + [d]) > best_max, so this branch is worse.
        # So we can prune d >= best_max if we are looking for a better solution.
        # But we don't know if we will find a better one.
        # So we can safely prune d >= best_max.
        
        max_d_to_try = best_max - 1 if best_max != float('inf') else min_d + 100000
        
        # If best_max is inf, we need to find *some* solution.
        # Greedy usually finds one. Let's use Greedy to get an initial best_max.
        
        if best_max == float('inf'):
            # Run greedy to get an initial solution
            r = Fraction(num, den)
            greedy_sol = []
            g_used = set(used)
            while r and len(greedy_sol) < 256:
                d = (r.denominator + r.numerator - 1) // r.numerator
                if d in g_used:
                    d += 1
                    while d in g_used:
                        d += 1
                if d > 10000000:
                    break
                greedy_sol.append(d)
                g_used.add(d)
                r -= Fraction(1, d)
            if r == 0 and greedy_sol:
                full_sol = path + greedy_sol
                update_best(full_sol)
            # If we still have inf, we might fail, but greedy usually works for 5/31.
            if best_max == float('inf'):
                return
                
        # Now we have a best_max. We can prune d >= best_max.
        
        # Iterate d
        for d in range(min_d, max_d_to_try + 1):
            if d in used:
                continue
                
            # Check if d is valid
            # New remainder: num/den - 1/d = (num*d - den) / (den*d)
            new_num = num * d - den
            new_den = den * d
            
            if new_num < 0:
                continue # Should not happen since d > den/num
            
            if new_num == 0:
                # Found a solution
                new_path = path + [d]
                update_best(new_path)
                return # No need to check other d for this state, as we found a complete solution
                
            # Prune if new_num is too large? No.
            # Prune if new_den is too large? No, but it grows.
            
            # Recurse
            # To minimize max, we should try smaller d first.
            # We are iterating d upwards, so this is correct.
            
            # Additional pruning:
            # If new_num/new_den > 1/(min_possible_next_d) ...
            
            # Call dfs
            # We pass a copy of used and path
            # To save memory/time, we can mutate and backtrack, but sets are small.
            new_used = used | {d}
            new_path = path + [d]
            
            # Heuristic: If new_num is small, the next denominator will be large.
            # If new_num is large, the next denominator will be small.
            
            dfs(new_num, new_den, new_used, new_path, max(current_max, d))
            
            # If we found a solution with max < current best, we might want to break?
            # No, we want to find the MINIMUM max.
            # So we must continue searching for smaller max.
            # But if we found a solution, best_max is updated.
            # The loop condition `d < max_d_to_try` uses the old max_d_to_try.
            # We should update max_d_to_try dynamically?
            # Or just let it run. The pruning `d >= best_max` in the next iteration will handle it.
            # Actually, `max_d_to_try` is calculated before the loop.
            # If best_max decreases, we should stop early.
            if d >= best_max:
                break

    # Start search
    # Initial state: 5/31, used={}, path=[]
    dfs(5, 31, set(), [], 0)
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: Greedy
    r = target
    values = []
    used = set()
    while r and len(values) < 256:
        d = (r.denominator + r.numerator - 1) // r.numerator
        if d in used:
            d += 1
            while d in used:
                d += 1
        if d > 10000000:
            break
        values.append(d)
        used.add(d)
        r -= Fraction(1, d)
        
    if r == 0 and values:
        return values
        
    # Absolute fallback (should not happen)
    return [8, 17, 4216]

if __name__ == "__main__":
    sol = construct()
    s = sum(Fraction(1, d) for d in sol)
    print(f"Solution: {sol}")
    print(f"Sum: {s} (target 5/31 = {Fraction(5,31)})")
    print(f"Max denominator: {max(sol)}")
    print(f"Count: {len(sol)}")
    print(f"Valid: {s == Fraction(5,31) and len(set(sol))==len(sol) and all(1<=d<=10000000 for d in sol) and len(sol)<=256}")

