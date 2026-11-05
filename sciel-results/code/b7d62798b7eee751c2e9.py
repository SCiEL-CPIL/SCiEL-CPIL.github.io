import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    target = Fraction(5, 31)
    
    def dfs(r, k, min_d, max_d, current_list):
        if time.time() > deadline:
            return None, False
        
        if k == 0:
            if r == 0:
                return list(current_list), True
            else:
                return None, False
        
        if k == 1:
            if r.numerator == 1:
                d = r.denominator
                if min_d <= d <= max_d and d not in current_list:
                    return current_list + [d], True
            return None, False
            
        # d must satisfy 1/d <= r  => d >= 1/r
        # and k/d >= r => d <= k/r
        min_possible_d = max(min_d, (r.denominator + r.numerator - 1) // r.numerator)
        max_possible_d = min(max_d, (k * r.denominator) // r.numerator)
        
        if min_possible_d > max_possible_d:
            return None, False
            
        for d in range(min_possible_d, max_possible_d + 1):
            if d in current_list:
                continue
            new_r = r - Fraction(1, d)
            if new_r < 0:
                break
            res, found = dfs(new_r, k - 1, d + 1, max_d, current_list + [d])
            if found:
                return res, True
                
        return None, False

    # We want to minimize the maximum denominator.
    # We know a solution exists with max denom 1736.
    # Let's binary search for the minimum max denominator M.
    low = 1
    high = 1736
    
    best_solution = None
    
    # First, ensure we have a valid solution within time.
    # Try to find a solution with M=1736 to have a fallback.
    sol_1736, found_1736 = dfs(target, 10, 1, 1736, [])
    if found_1736:
        best_solution = sol_1736
    
    # Binary search for smaller M
    while low < high and time.time() < deadline:
        mid = (low + high) // 2
        sol_mid, found_mid = dfs(target, 10, 1, mid, [])
        if found_mid:
            best_solution = sol_mid
            high = mid
        else:
            low = mid + 1
            
    if best_solution is None:
        # Fallback to greedy if DFS fails or times out
        r, values = target - Fraction(1, 8), [8]
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
        
    return best_solution
