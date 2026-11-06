import time
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s limit
    
    best_solution = [8, 28, 1736]
    best_max_denom = 1736
    
    # We will use a global set for memoization within a single find_solution call
    # But find_solution is called multiple times with different limits.
    # So we need to clear memoization for each limit?
    # Or keep it? If a state is unsolvable for limit K, it is unsolvable for K-1.
    # So we can keep a global "unsolvable" set.
    # But "solvable" depends on the limit.
    # So we can't easily memoize "solvable".
    # We can memoize "unsolvable" for a specific limit?
    # No, if unsolvable for K, unsolvable for K-1.
    # So we can maintain a set of states that are known to be unsolvable for the *current* limit.
    # But the limit changes.
    
    # Better: Just use lru_cache inside dfs?
    # But dfs depends on `limit` and `H`.
    # So we define dfs inside find_solution.
    
    def find_solution(limit):
        H = [0.0] * (limit + 1)
        for i in range(1, limit + 1):
            H[i] = H[i-1] + 1.0/i
        
        # Memoization for unsolvable states in this specific limit
        # Key: (n, d, min_d)
        # Value: None (just a set of keys)
        unsolvable = set()
        
        def dfs(n, d, min_d, path):
            if n == 0:
                return path
            if n < 0:
                return None
            
            state = (n, d, min_d)
            if state in unsolvable:
                return None
            
            lower = max(min_d, (d + n - 1) // n)
            
            if lower > limit:
                unsolvable.add(state)
                return None
            
            found_solution = False
            for x in range(lower, limit + 1):
                if time.time() - start_time > time_limit:
                    return None
                
                num_next = n * x - d
                if num_next == 0:
                    return path + [x]
                if num_next < 0:
                    break
                
                den_next = d * x
                g = gcd(num_next, den_next)
                if g > 1:
                    num_next //= g
                    den_next //= g
                
                # Pruning
                # Max sum available from x+1 to limit
                if H[limit] - H[x] < num_next / den_next + 1e-12:
                    # If even the max possible sum is less than required, break
                    # Since H[limit]-H[x] decreases and num_next/den_next increases with x?
                    # num_next = n*x - d. As x increases, num_next increases.
                    # den_next = d*x. As x increases, den_next increases.
                    # ratio = (nx-d)/dx = n/d - 1/x.
                    # As x increases, 1/x decreases, so ratio increases.
                    # H[limit]-H[x] decreases.
                    # So if it fails, it will fail for all larger x.
                    break
                
                res = dfs(num_next, den_next, x + 1, path + [x])
                if res is not None:
                    return res
            
            unsolvable.add(state)
            return None
        
        return dfs(5, 31, 1, [])
    
    current_limit = best_max_denom - 1
    while current_limit >= 1:
        if time.time() - start_time > time_limit - 1.0:
            break
        sol = find_solution(current_limit)
        if sol:
            actual_max = max(sol)
            if actual_max < best_max_denom:
                best_max_denom = actual_max
                best_solution = sol
            current_limit = actual_max - 1
        else:
            break
            
    return best_solution
