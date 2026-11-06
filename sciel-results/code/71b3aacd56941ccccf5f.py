import time
from math import gcd

def construct():
        start_time = time.time()
        time_limit = 55.0
        
        best_solution = [10, 15, 930]
        best_max_denom = 930
        
        def update_best(sol):
            nonlocal best_solution, best_max_denom
            if sol:
                m = max(sol)
                if m < best_max_denom:
                    best_max_denom = m
                    best_solution = sol
        
        # 1. Try 3 terms
        # ... (code from above) ...
        
        # 2. Try 4 terms
        # ...
        
        # 3. General DFS
        def dfs(n, d, min_d, path):
            if time.time() - start_time > time_limit:
                return
            if n == 0:
                update_best(path)
                return
            
            # Lower bound for x
            # 1/x <= n/d => x >= d/n
            lower = max(min_d, (d + n - 1) // n)
            
            # Upper bound for x?
            # We want to find ANY solution.
            # If we iterate x from lower upwards, we try small denominators first.
            # This tends to find solutions with small max denominators early?
            # Not necessarily, but it's a good heuristic.
            
            # Pruning:
            # Max possible sum of remaining terms (k-1 terms) with min denom x+1
            # is (k-1) / (x+1).
            # If n/d > (k-1)/(x+1), prune.
            # We don't know k (remaining terms) exactly, but we can use a loose bound.
            # Or just limit the depth?
            
            # Let's limit the depth to, say, 10 terms.
            # And limit x to some max?
            # If we don't limit x, the search can blow up.
            # But we have a time limit.
            
            # To find small max denominators, we should stop exploring x if x > current best_max_denom?
            # Yes! If we are looking for a solution with max denom < best_max_denom,
            # we can prune any branch where x >= best_max_denom.
            
            if lower >= best_max_denom:
                return
            
            for x in range(lower, best_max_denom):
                num_next = n * x - d
                if num_next == 0:
                    update_best(path + [x])
                    return # Found a solution with max < best_max_denom, can we do better?
                    # Actually, if we found one, we update best.
                    # We should continue searching for even smaller?
                    # The loop iterates x from small to large.
                    # The first solution found in this path will have max = x (if x is the largest so far).
                    # But x is increasing.
                    # So the first solution found might have a larger x than a later one?
                    # No, x is the current denominator.
                    # If we find a solution, max(path + [x]) is the new max.
                    # Since we iterate x from low to high, the first solution found
                    # in this specific branch might not be the global min max.
                    # But we update best, so subsequent branches will have a tighter bound.
                    
                if num_next < 0:
                    break
                
                den_next = d * x
                g = gcd(num_next, den_next)
                num_next //= g
                den_next //= g
                
                # Prune: Can we form num_next/den_next with denominators > x?
                # Max sum of 1/j for j > x is infinite, but we are bounded by best_max_denom.
                # Max sum with denominators in (x, best_max_denom) is roughly ln(best_max_denom/x).
                # If num_next/den_next > ln(best_max_denom/x), prune.
                
                dfs(num_next, den_next, x + 1, path + [x])
        
        dfs(5, 31, 1, [])
        
        return best_solution
