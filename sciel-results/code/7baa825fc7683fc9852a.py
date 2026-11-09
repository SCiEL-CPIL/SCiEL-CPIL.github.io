import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Try to find a solution with k terms, starting from k=5
    for k in range(5, 11):
        solution = find_solution(k, start_time, deadline)
        if solution:
            return solution
            
    # Fallback: return a known valid solution if search fails
    # Example: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + ...
    # But we must minimize count. If search fails, we return the best found.
    # If no solution found, raise error or return a large set?
    # The task requires 1..256 distinct denominators.
    # If we can't find a small one, we can construct one.
    # Greedy algorithm will always find one.
    return greedy_solution(start_time, deadline)

def find_solution(k, start_time, deadline):
    # DFS to find k distinct denominators >= 5 summing to 1
    # d[0] >= 5
    # d[i] > d[i-1]
    
    def dfs(idx, last_d, rem_frac, current_list):
        if time.time() > deadline:
            return None
            
        if idx == k:
            if rem_frac == 0:
                return current_list
            else:
                return None
        
        # If we have 1 term left, check if rem_frac is a unit fraction
        if idx == k - 1:
            # rem_frac = 1/d
            if rem_frac.numerator == 1:
                d = rem_frac.denominator
                if d > last_d and d <= 10000000:
                    return current_list + [d]
            return None
            
        # Bounds for next denominator d
        # 1/d < rem_frac  => d > 1/rem_frac
        # Also, we need to be able to sum (k-idx) terms starting from d
        # Max sum of (k-idx) terms with smallest possible denominators d, d+1, ...
        # Min sum of (k-idx) terms with largest possible denominators? No, denominators are bounded by 10^7.
        # Actually, we just need sum >= rem_frac.
        
        min_d = max(last_d + 1, 5)
        
        # Upper bound:
        # 1/d + 1/(d+1) + ... + 1/(d + m - 1) >= rem_frac
        # Approx 1/d * m >= rem_frac => d <= m / rem_frac
        # More precisely, sum_{i=0}^{m-1} 1/(d+i) >= rem_frac
        # This gives an upper bound on d.
        
        m = k - idx
        # d must be such that the sum of m unit fractions starting from d is >= rem_frac
        # And 1/d < rem_frac (strictly, because we need m terms, if 1/d == rem_frac, m must be 1, handled above)
        
        # Lower bound for d:
        # 1/d < rem_frac => d > 1/rem_frac
        # d_min = floor(1/rem_frac) + 1
        d_min = max(min_d, rem_frac.denominator // rem_frac.numerator + 1)
        
        # Upper bound for d:
        # We need sum_{i=0}^{m-1} 1/(d+i) >= rem_frac
        # Let's find max d such that this holds.
        # 1/d + 1/(d+1) + ... >= rem_frac
        # For large d, sum ~ m/d. So d ~ m/rem_frac.
        # Let's start from a reasonable upper bound and go down? Or up?
        # d must be small enough.
        # d_max such that sum_{i=0}^{m-1} 1/(d_max+i) >= rem_frac
        # If d is too large, sum is too small.
        
        # Estimate d_max
        # m/d_max >= rem_frac => d_max <= m / rem_frac
        d_max = int(m / rem_frac) + 10
        # Ensure d_max is not too huge
        if d_max > 10000000:
            d_max = 10000000
            
        # Iterate d from d_min to d_max
        # Optimization: if d_min > d_max, no solution
        if d_min > d_max:
            return None
            
        for d in range(d_min, d_max + 1):
            new_rem = rem_frac - Fraction(1, d)
            if new_rem < 0:
                break # d is too small? No, if 1/d > rem, new_rem < 0. 
                # But we ensured d > 1/rem, so 1/d < rem. So new_rem > 0.
                # Wait, d_min > 1/rem. So 1/d < rem. So new_rem > 0.
                
            # Recursive call
            res = dfs(idx + 1, d, new_rem, current_list + [d])
            if res:
                return res
                
        return None

    return dfs(0, 4, Fraction(1, 1), [])

def greedy_solution(start_time, deadline):
    # Fallback: Greedy algorithm
    rem = Fraction(1, 1)
    last_d = 4
    result = []
    
    while rem > 0 and time.time() < deadline:
        # Next denominator
        # 1/d < rem => d > 1/rem
        d = rem.denominator // rem.numerator + 1
        if d <= last_d:
            d = last_d + 1
            
        # Ensure d >= 5
        if d < 5:
            d = 5
            
        result.append(d)
        rem -= Fraction(1, d)
        last_d = d
        
        # If rem becomes 0, we are done
        if rem == 0:
            break
            
        # If we exceed 256 terms, break (should not happen for greedy on 1)
        if len(result) > 256:
            break
            
    return result
