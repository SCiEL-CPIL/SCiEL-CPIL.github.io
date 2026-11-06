import time
from math import gcd, log
import heapq

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    # Known good solution
    best_solution = [10, 15, 930]
    best_max_denom = 930
    
    def dijkstra_search(max_denom_limit):
        """
        Find solution with all denominators <= max_denom_limit
        Returns list of denominators or None
        """
        pq = []
        heapq.heappush(pq, (1, 5, 31, 1, []))
        best_cost = {}
        
        while pq:
            if time.time() - start_time > time_limit:
                return None
            
            cost, n, d, min_d, path = heapq.heappop(pq)
            
            state_key = (n, d, min_d)
            if state_key in best_cost and best_cost[state_key] <= cost:
                continue
            best_cost[state_key] = cost
            
            if n == 0:
                return path
            
            if n <= 0:
                continue
            
            lower = max(min_d, (d + n - 1) // n)
            
            if lower > max_denom_limit:
                continue
            
            for x in range(lower, max_denom_limit + 1):
                new_cost = max(cost, x)
                if new_cost >= best_max_denom:
                    break
                
                num_next = n * x - d
                if num_next < 0:
                    break
                
                if num_next == 0:
                    return path + [x]
                
                den_next = d * x
                g = gcd(num_next, den_next)
                num_next //= g
                den_next //= g
                
                # Prune: max possible sum from x+1 to max_denom_limit
                if x < max_denom_limit:
                    if log(max_denom_limit / x) < num_next / den_next:
                        break
                
                new_state_key = (num_next, den_next, x + 1)
                if new_state_key in best_cost and best_cost[new_state_key] <= new_cost:
                    continue
                
                heapq.heappush(pq, (new_cost, num_next, den_next, x + 1, path + [x]))
        
        return None
    
    # Try increasing max_denom limits
    for max_d in [200, 300, 400, 500, 700, 900]:
        sol = dijkstra_search(max_d)
        if sol:
            actual_max = max(sol)
            if actual_max < best_max_denom:
                best_max_denom = actual_max
                best_solution = sol
                break
    
    return best_solution
