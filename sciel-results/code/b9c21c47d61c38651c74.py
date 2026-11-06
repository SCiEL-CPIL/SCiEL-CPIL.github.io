import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    target = Fraction(5, 31)
    
    # Known valid solution: 1/7 + 1/56 + 1/1736 = 5/31
    best_solution = [7, 56, 1736]
    best_max = 1736
    
    # Precompute harmonic numbers as floats for pruning
    # We search up to 1736
    max_D_search = 1736
    H = [0.0] * (max_D_search + 2)
    for i in range(1, max_D_search + 1):
        H[i] = H[i-1] + 1.0/i
        
    def dfs(R, d_prev, D, path):
        if time.time() > deadline:
            return None
            
        if R == 0:
            return list(path)
            
        if R < 0:
            return None
            
        if len(path) >= 256:
            return None
            
        # Prune: Max possible sum with remaining denominators
        # Denominators must be > d_prev and <= D
        max_sum = H[D] - H[d_prev]
        # If R > max_sum, impossible
        # Use float for speed, with epsilon
        if float(R) > max_sum + 1e-12:
            return None
            
        # Determine d_min
        # We need 1/d < R => d > 1/R
        # Also d > d_prev
        # Also d <= D
        
        # Check if R is a unit fraction 1/d
        if R.numerator == 1:
            d_exact = R.denominator
            if d_exact > d_prev and d_exact <= D:
                path.append(d_exact)
                return list(path)
                
        # d_min from 1/R
        # d > 1/R => d >= floor(1/R) + 1
        inv_R = 1.0 / float(R)
        d_min_float = int(inv_R) + 1
        
        d_min = max(d_prev + 1, d_min_float)
        
        if d_min > D:
            return None
            
        for d in range(d_min, D + 1):
            new_R = R - Fraction(1, d)
            if new_R < 0:
                # This shouldn't happen if d_min is correct, but just in case
                # If 1/d > R, then d < 1/R. 
                # But d >= d_min > 1/R. So 1/d < R.
                # So new_R > 0.
                continue
                
            path.append(d)
            res = dfs(new_R, d, D, path)
            if res is not None:
                return res
            path.pop()
            
        return None

    # Binary search for smallest D
    lo = 10
    hi = best_max
    
    while lo < hi:
        if time.time() > deadline - 5.0:
            break
        mid = (lo + hi) // 2
        # Check if solution exists with max denom <= mid
        res = dfs(target, 0, mid, [])
        if res is not None:
            actual_max = max(res)
            if actual_max < best_max:
                best_max = actual_max
                best_solution = res
            hi = mid
        else:
            lo = mid + 1
            
    return best_solution
