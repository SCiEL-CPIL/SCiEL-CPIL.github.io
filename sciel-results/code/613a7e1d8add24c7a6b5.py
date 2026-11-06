import time
import math
from math import gcd
import heapq

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    # Known good solution
    best_solution = [10, 15, 930]
    best_max_denom = 930
    
    def dijkstra_search(max_denom):
        # Dijkstra on (n, d, min_d) with cost = max denominator in path
        # Use early termination when we find any valid solution
        
        parent = {}
        best_cost = {}
        pq = []
        
        start_state = (5, 31, 1)
        heapq.heappush(pq, (1, start_state))
        best_cost[start_state] = 1
        
        while pq:
            if time.time() - start_time > time_limit:
                break
            
            curr_max, state = heapq.heappop(pq)
            n, d, min_d = state
            
            if state in best_cost and best_cost[state] < curr_max:
                continue
            
            if n == 0:
                # Reconstruct path
                path = []
                cur = state
                while cur in parent:
                    prev, x = parent[cur]
                    path.append(x)
                    cur = prev
                path.reverse()
                return path
            
            if n < 0:
                continue
            
            lower = max(min_d, (d + n - 1) // n)
            if lower > max_denom:
                continue
            
            for x in range(lower, max_denom + 1):
                num_next = n * x - d
                if num_next == 0:
                    new_max = max(curr_max, x)
                    term_state = (0, 1, x + 1)
                    if term_state not in best_cost or best_cost[term_state] > new_max:
                        best_cost[term_state] = new_max
                        parent[term_state] = (state, x)
                        heapq.heappush(pq, (new_max, term_state))
                    continue
                    
                if num_next < 0:
                    break
                
                den_next = d * x
                g = gcd(num_next, den_next)
                num_next //= g
                den_next //= g
                
                new_max = max(curr_max, x)
                next_state = (num_next, den_next, x + 1)
                
                if next_state in best_cost and best_cost[next_state] <= new_max:
                    continue
                
                # Prune: upper bound on remaining sum
                if x + 1 <= max_denom:
                    upper_bound_sum = (max_denom - x) / (x + 1)
                    if upper_bound_sum < num_next / den_next:
                        break
                else:
                    break
                
                best_cost[next_state] = new_max
                parent[next_state] = (state, x)
                heapq.heappush(pq, (new_max, next_state))
        
        return None
    
    for max_d in [200, 300, 400, 500, 600, 800, 1000]:
        sol = dijkstra_search(max_d)
        if sol:
            actual_max = max(sol)
            if actual_max < best_max_denom:
                best_max_denom = actual_max
                best_solution = sol
                break
    
    return best_solution
