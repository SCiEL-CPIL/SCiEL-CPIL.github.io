"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s safety margin
    
    target = Fraction(5, 31)
    
    # 1. Find an initial valid solution to set the upper bound for binary search
    # Using a simple greedy method to get a valid (though not optimal) solution
    def greedy_solution(r, used):
        if r == 0:
            return []
        if r < 0:
            return None
        values = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d > 10000000:
                return None
            if d in used:
                # If d is used, try d+1, d+2...
                found = False
                for dd in range(d, min(d + 100, 10000001)):
                    if dd not in used:
                        new_r = r - Fraction(1, dd)
                        if new_r >= 0:
                            values.append(dd)
                            used.add(dd)
                            r = new_r
                            found = True
                            break
                if not found:
                    return None
            else:
                values.append(d)
                used.add(d)
                r -= Fraction(1, d)
        return values

    initial_solution = greedy_solution(target, set())
    if initial_solution is None:
        # Fallback if greedy fails (unlikely for 5/31)
        return [8, 28, 1736] # Known solution from inspiration/seed
    
    best_solution = initial_solution
    best_max_den = max(initial_solution)
    
    # 2. Binary search for the minimal maximum denominator K
    # We know a solution exists for K = best_max_den
    # We want the smallest K such that a solution exists with all denoms <= K
    
    low = 1
    high = best_max_den
    
    # Helper: Check if a solution exists with max denom <= K
    def is_feasible(K):
        # Use DFS with pruning
        # State: remainder, min_denom (next denom must be > min_denom to ensure distinctness and order)
        # Actually, to ensure distinctness, we just enforce d > previous_d.
        
        def dfs(remainder, min_denom, current_path):
            nonlocal best_solution, best_max_den
            
            if time.time() > deadline:
                return False
            
            if remainder == 0:
                # Found a valid solution
                # The max denominator in this path is max(current_path)
                # But we are constrained by K, so max(current_path) <= K
                # We update the global best if this solution has a smaller max denom
                # Note: Since we are binary searching K, any solution found here has max <= K.
                # To minimize the *actual* max denom, we should track the real max.
                actual_max = max(current_path) if current_path else 0
                if actual_max < best_max_den:
                    best_max_den = actual_max
                    best_solution = current_path[:]
                return True
            
            if remainder < 0:
                return False
            
            # Pruning 1: If remainder is too small to be formed by any single term <= K
            # Smallest term is 1/K. If remainder < 1/K, we can't subtract any valid term.
            if remainder < Fraction(1, K):
                return False
            
            # Pruning 2: Max possible sum check
            # Sum of 1/d for d in [min_denom+1, K]
            # Approximation: ln(K) - ln(min_denom)
            # If this max possible sum < remainder, infeasible.
            if min_denom >= K:
                return False
                
            # Use harmonic approximation for speed
            # H_K - H_min_denom >= remainder?
            # H_n ~ ln(n) + gamma
            # We can use a tighter bound or just the log approx.
            # Let's use the log approx. It's an overestimate of the sum?
            # Sum(1/i) from a to b is approx ln(b/a).
            # Actually Sum(1/i) > ln((b+1)/a).
            # Let's just check if ln(K / min_denom) < float(remainder).
            # If so, even taking all available denominators, we can't reach remainder.
            if min_denom > 0:
                max_sum_approx = math.log(K) - math.log(min_denom)
                if max_sum_approx < float(remainder):
                    return False
            
            # Determine range for next denominator d
            # 1. d > min_denom
            # 2. 1/d <= remainder => d >= 1/remainder
            lower_bound = (1 / remainder).numerator // (1 / remainder).denominator
            if lower_bound < min_denom + 1:
                lower_bound = min_denom + 1
            
            if lower_bound > K:
                return False
            
            # Iterate d
            # Heuristic: Try smaller d first?
            # Smaller d -> larger term -> remainder drops faster.
            # This tends to find solutions with fewer terms or larger terms.
            # For minimizing the *largest* denominator, we are already bounding by K.
            # Finding *any* solution with max <= K is the goal of the feasibility check.
            
            for d in range(lower_bound, K + 1):
                if time.time() > deadline:
                    return False
                
                next_rem = remainder - Fraction(1, d)
                
                if next_rem < 0:
                    # Since d is increasing, 1/d is decreasing.
                    # next_rem = r - 1/d will increase as d increases.
                    # So if next_rem < 0, it might become >= 0 for larger d?
                    # No, wait.
                    # r is fixed. 1/d decreases as d increases.
                    # So r - 1/d increases as d increases.
                    # If r - 1/d < 0, then 1/d > r.
                    # As d increases, 1/d gets smaller, so eventually 1/d <= r.
                    # So we should CONTINUE, not break.
                    # However, we established lower_bound such that 1/d <= r.
                    # So next_rem should be >= 0.
                    # If it is negative, it means our lower_bound calculation was slightly off or precision issue.
                    continue
                
                if next_rem == 0:
                    # Solution found
                    actual_max = max(current_path + [d])
                    if actual_max < best_max_den:
                        best_max_den = actual_max
                        best_solution = current_path + [d]
                    return True
                
                # Recurse
                if dfs(next_rem, d, current_path + [d]):
                    return True
            
            return False
        
        # Start DFS
        # min_denom starts at 0 (so first d can be 1)
        # But 1/1 = 1 > 5/31, so it will be skipped by lower_bound logic.
        return dfs(target, 0, [])

    # Binary Search
    # We know is_feasible(high) is True.
    # We want the lowest K where is_feasible(K) is True.
    # Note: is_feasible(K) might be expensive if infeasible.
    # But binary search reduces the number of calls to O(log N).
    
    # To speed up, we can check midpoints.
    # If is_feasible(mid) is True, high = mid.
    # If False, low = mid + 1.
    
    # However, the DFS might timeout. We treat timeout as "Infeasible" for the purpose of the binary search step?
    # No, if it times out, we don't know. 
    # Better strategy: 
    # 1. Try to improve the bound by checking specific values?
    # 2. Or just run the binary search, but if a check times out, we might get stuck.
    
    # Let's use a hybrid:
    # We have a current best_max_den.
    # We try to find a smaller K.
    # We can iterate K downwards from best_max_den? No, that's linear.
    # Binary search is better.
    
    # To handle timeouts:
    # If is_feasible(K) times out, we assume it's likely infeasible (or hard to find) and move low up?
    # Actually, if it's feasible but hard to find, we might miss it.
    # But the DFS is quite efficient with pruning.
    
    # Let's run binary search.
    # To ensure we don't waste time on a "False" that is actually a timeout,
    # we can limit the time per DFS call?
    # The global deadline is the hard stop.
    
    # Let's just do the binary search.
    while low < high:
        if time.time() > deadline - 2: # Leave 2s for final return
            break
            
        mid = (low + high) // 2
        # We need to know if mid is feasible.
        # The is_feasible function updates best_solution if it finds a better one.
        # But for the binary search logic, we need a boolean.
        # Let's modify is_feasible to return True/False/None (timeout).
        
        def check_feasible(K):
            nonlocal best_solution, best_max_den
            # We need to reset the "found" state for this specific K check?
            # No, we want to keep the global best.
            # But we need to know if *this* K allows a solution.
            # If a solution exists for K, does it exist for K+1? Yes.
            # So the set of feasible K is an interval [K_min, 10^7].
            
            # We run DFS for this K.
            # If it finds a solution, return True.
            # If it exhausts search, return False.
            # If it times out, return None.
            
            start_check_time = time.time()
            
            def dfs_check(remainder, min_denom, current_path):
                nonlocal best_solution, best_max_den
                
                if time.time() > deadline - 1: # Hard stop
                    return "timeout"
                
                if remainder == 0:
                    actual_max = max(current_path) if current_path else 0
                    if actual_max < best_max_den:
                        best_max_den = actual_max
                        best_solution = current_path[:]
                    return True
                
                if remainder < 0:
                    return False
                
                if remainder < Fraction(1, K):
                    return False
                
                if min_denom >= K:
                    return False
                
                if min_denom > 0:
                    max_sum_approx = math.log(K) - math.log(min_denom)
                    if max_sum_approx < float(remainder):
                        return False
                
                lower_bound = (1 / remainder).numerator // (1 / remainder).denominator
                if lower_bound < min_denom + 1:
                    lower_bound = min_denom + 1
                
                if lower_bound > K:
                    return False
                
                for d in range(lower_bound, K + 1):
                    if time.time() > deadline - 1:
                        return "timeout"
                    
                    next_rem = remainder - Fraction(1, d)
                    
                    if next_rem < 0:
                        continue
                    
                    if next_rem == 0:
                        actual_max = max(current_path + [d])
                        if actual_max < best_max_den:
                            best_max_den = actual_max
                            best_solution = current_path + [d]
                        return True
                    
                    res = dfs_check(next_rem, d, current_path + [d])
                    if res == True:
                        return True
                    if res == "timeout":
                        return "timeout"
                
                return False
            
            return dfs_check(target, 0, [])

        res = check_feasible(mid)
        
        if res == True:
            high = mid
        elif res == False:
            low = mid + 1
        else: # timeout
            # If timeout, we don't know. 
            # It's safer to assume it might be feasible? 
            # If we assume feasible, high = mid. We might shrink the range too fast if it's actually infeasible.
            # If we assume infeasible, low = mid + 1. We might skip the answer.
            # Given the pruning, timeouts are likely in hard-to-find-feasible cases (large K, complex remainder) 
            # or infeasible cases with poor pruning.
            # However, for small K (near the optimum), solutions are usually found quickly if they exist.
            # Let's assume infeasible (low = mid + 1) to avoid getting stuck on a large infeasible range?
            # Actually, if it's infeasible, the search space is empty, so it should finish quickly?
            # No, "empty" search space might still traverse many nodes before proving empty.
            # Let's try low = mid + 1.
            low = mid + 1

    return best_solution

# EVOLVE-BLOCK-END
