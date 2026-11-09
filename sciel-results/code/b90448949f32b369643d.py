import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit
    
    # Try to find a solution with k terms, starting from k=9
    # Max sum for k=8 is < 1, so k must be >= 9
    for k in range(9, 13):
        solution = find_solution(k, start_time, deadline)
        if solution:
            return solution
            
    # Fallback: Greedy algorithm
    return greedy_solution(start_time, deadline)

def find_solution(k, start_time, deadline):
    def dfs(idx, last_d, rem_frac, current_list):
        if time.time() > deadline:
            return None
            
        if idx == k:
            if rem_frac == 0:
                return current_list
            else:
                return None
        
        if idx == k - 1:
            if rem_frac.numerator == 1:
                d = rem_frac.denominator
                if d > last_d and d <= 10000000:
                    return current_list + [d]
            return None
            
        m = k - idx
        d_min = max(last_d + 1, 5, rem_frac.denominator // rem_frac.numerator + 1)
        
        # Upper bound: sum of m terms starting from d_min must be >= rem
        # Approx d_max = m / rem_frac
        d_max = int(m / rem_frac) + 10
        if d_max > 10000000:
            d_max = 10000000
            
        if d_min > d_max:
            return None
            
        for d in range(d_min, d_max + 1):
            new_rem = rem_frac - Fraction(1, d)
            if new_rem < 0:
                break
            res = dfs(idx + 1, d, new_rem, current_list + [d])
            if res:
                return res
                
        return None

    return dfs(0, 4, Fraction(1, 1), [])

def greedy_solution(start_time, deadline):
    rem = Fraction(1, 1)
    last_d = 4
    result = []
    
    while rem > 0 and time.time() < deadline:
        # Next denominator
        # 1/d < rem => d > 1/rem
        # d must be > last_d and >= 5
        d = max(last_d + 1, 5, rem.denominator // rem.numerator + 1)
        
        if d > 10000000:
            break
            
        result.append(d)
        rem -= Fraction(1, d)
        last_d = d
        
        if rem == 0:
            break
            
        if len(result) > 256:
            break
            
    return result
