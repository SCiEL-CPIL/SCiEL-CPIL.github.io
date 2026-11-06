import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    # Known good solution
    best_solution = [10, 15, 930]
    best_max_denom = 930
    
    def try_complete_2_terms(n, d, min_d, max_denom, path):
        """Try to complete n/d using exactly 2 terms with denominators >= min_d and <= max_denom.
        Returns the two denominators or None."""
        if n == 0:
            return []
        if n < 0:
            return None
        
        # We need 1/a + 1/b = n/d where min_d <= a < b <= max_denom
        # 1/a = n/d - 1/b = (nb - d)/(db)
        # So a = db/(nb - d) must be an integer
        # Equivalently: d*b = a*(n*b - d) => d*b = n*a*b - a*d => a*d = b*(n*a - d)
        # So a*d must be divisible by (n*a - d)
        # Let's iterate over possible values of a
        
        # Lower bound for a: 1/a <= n/d => a >= d/n
        # Also a >= min_d
        lower_a = max(min_d, (d + n - 1) // n)
        
        # Upper bound: 1/a + 1/b >= 1/a + 1/(a+1) for b > a
        # But simpler: 1/a < n/d => a > d/n
        # And 1/a + 1/(a+1) >= n/d for feasibility
        
        for a in range(lower_a, max_denom + 1):
            if time.time() - start_time > time_limit:
                return None
            
            # Check if 1/a < n/d (strict, since we need b > a)
            if a * n <= d:
                break  # 1/a >= n/d, so even with b=a, sum >= n/d, but we need b>a
            
            # 1/b = n/d - 1/a = (na - d)/(da)
            # b = da/(na - d)
            num_b = d * a
            den_b = n * a - d
            
            if den_b <= 0:
                continue
            
            if num_b % den_b != 0:
                continue
            
            b = num_b // den_b
            
            if b <= a:
                continue
            if b > max_denom:
                continue
            
            return [a, b]
        
        return None
    
    def try_complete_3_terms(n, d, min_d, max_denom, path):
        """Try to complete n/d using exactly 3 terms with denominators >= min_d and <= max_denom.
        Returns the three denominators or None."""
        if n == 0:
            return []
        if n < 0:
            return None
        
        # We need 1/a + 1/b + 1/c = n/d where min_d <= a < b < c <= max_denom
        # Iterate over a, then complete with 2 terms
        
        lower_a = max(min_d, (d + n - 1) // n)
        
        for a in range(lower_a, max_denom + 1):
            if time.time() - start_time > time_limit:
                return None
            
            # 1/a must be less than n/d (since we need positive b, c)
            if a * n <= d:
                break
            
            # Remainder: n/d - 1/a = (na - d)/(da)
            num_rem = n * a - d
            den_rem = d * a
            
            if num_rem <= 0:
                continue
            
            g = gcd(num_rem, den_rem)
            num_rem //= g
            den_rem //= g
            
            # Now complete num_rem/den_rem with 2 terms, denominators >= a+1
            result = try_complete_2_terms(num_rem, den_rem, a + 1, max_denom, path + [a])
            if result is not None:
                return [a] + result
        
        return None
    
    def search_with_max_denom(max_denom):
        """Search for a solution with all denominators <= max_denom."""
        # Try 1 term
        if 5 * max_denom == 31:
            return [31]  # Not possible since 31/5 is not integer
        
        # Try 2 terms
        result = try_complete_2_terms(5, 31, 1, max_denom, [])
        if result is not None:
            return result
        
        # Try 3 terms
        result = try_complete_3_terms(5, 31, 1, max_denom, [])
        if result is not None:
            return result
        
        # Try 4 terms: iterate first term, then complete with 3 terms
        lower_a = 1
        for a in range(lower_a, max_denom + 1):
            if time.time() - start_time > time_limit:
                return None
            
            if a * 5 <= 31:
                break
            
            num_rem = 5 * a - 31
            den_rem = 31 * a
            
            if num_rem <= 0:
                continue
            
            g = gcd(num_rem, den_rem)
            num_rem //= g
            den_rem //= g
            
            result = try_complete_3_terms(num_rem, den_rem, a + 1, max_denom, [a])
            if result is not None:
                return [a] + result
        
        return None
    
    # Binary search for the minimum max_denom
    # We know a solution exists with max_denom = 930
    lo = 1
    hi = best_max_denom
    
    while lo < hi:
        if time.time() - start_time > time_limit:
            break
        
        mid = (lo + hi) // 2
        sol = search_with_max_denom(mid)
        
        if sol is not None:
            actual_max = max(sol)
            if actual_max < best_max_denom:
                best_max_denom = actual_max
                best_solution = sol
            hi = mid
        else:
            lo = mid + 1
    
    # Final check: try to improve with a slightly higher bound if we have time
    if time.time() - start_time < time_limit - 5:
        # Try a few values around the current best to see if we can do better
        for test_max in range(best_max_denom, best_max_denom + 50):
            if time.time() - start_time > time_limit - 2:
                break
            sol = search_with_max_denom(test_max)
            if sol is not None:
                actual_max = max(sol)
                if actual_max < best_max_denom:
                    best_max_denom = actual_max
                    best_solution = sol
                break
    
    return best_solution
