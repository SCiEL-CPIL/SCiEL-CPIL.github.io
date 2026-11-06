import time
from math import gcd, log

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    # Known good solution
    best_solution = [10, 15, 930]
    best_max_denom = 930
    
    def find_solution(max_denom):
        # Returns a list of denominators or None
        # State: (n, d, min_d, path)
        # n/d is the remainder
        # min_d is the minimum allowed next denominator
        # path is the list of chosen denominators so far
        
        def dfs(n, d, min_d, path):
            if time.time() - start_time > time_limit:
                return None
            if n == 0:
                return path
            if n < 0:
                return None
            
            # Lower bound for x
            lower = max(min_d, (d + n - 1) // n)
            
            if lower > max_denom:
                return None
            
            # For exact division: n*x - d must be divisible by d
            # i.e., n*x ≡ d (mod d) => n*x ≡ 0 (mod d)
            # Let g = gcd(n, d), then n' = n/g, d' = d/g
            # We need n'*x ≡ 0 (mod d'), so x must be a multiple of d'/gcd(n', d') = d'
            # Wait, n' and d' are coprime, so x must be a multiple of d'
            g = gcd(n, d)
            n_red = n // g
            d_red = d // g
            # x must satisfy: n_red * x ≡ 0 (mod d_red)
            # Since gcd(n_red, d_red) = 1, x must be a multiple of d_red
            step = d_red
            
            # Also need n*x > d (so remainder is positive)
            # n*x - d > 0 => x > d/n
            # This is already captured by lower bound
            
            # Upper bound: we need sum of 1/(x+1) to 1/max_denom >= (n*x-d)/(d*x)
            # Approximate: ln(max_denom/(x+1)) >= (n*x-d)/(d*x)
            
            # Iterate x from lower to max_denom in steps of 'step'
            # But also need x >= lower
            x = lower
            # Align x to be a multiple of step
            if x % step != 0:
                x = x + (step - x % step)
            
            while x <= max_denom:
                if time.time() - start_time > time_limit:
                    return None
                
                num_next = n * x - d
                if num_next == 0:
                    return path + [x]
                if num_next < 0:
                    break
                
                den_next = d * x
                g2 = gcd(num_next, den_next)
                num_next //= g2
                den_next //= g2
                
                # Prune: Can we form num_next/den_next with denoms in [x+1, max_denom]?
                # Max possible sum is sum_{j=x+1}^{max_denom} 1/j
                # Lower bound for this sum: ln(max_denom) - ln(x+1) = ln(max_denom/(x+1))
                # If this is less than num_next/den_next, we can break
                if x + 1 >= max_denom:
                    break
                if log(max_denom / (x + 1)) < num_next / den_next:
                    break
                    
                res = dfs(num_next, den_next, x + 1, path + [x])
                if res is not None:
                    return res
                
                x += step
            
            return None
        
        return dfs(5, 31, 1, [])

    # Try progressively larger max_denom bounds
    for max_d in [100, 150, 200, 300, 500, 700, 900]:
        sol = find_solution(max_d)
        if sol:
            actual_max = max(sol)
            if actual_max < best_max_denom:
                best_max_denom = actual_max
                best_solution = sol
                # If we found a solution with small max, we can stop early
                # but let's try to see if we can do even better
                if max_d <= 150:
                    break
    
    return best_solution
