import math
import time
import bisect
from functools import lru_cache

def construct():
    start_time = time.time()
    N = 479001600  # 12!
    
    primes = [2, 3, 5, 7, 11]
    max_exp = [10, 5, 2, 1, 1]
    target_pf = tuple(max_exp)
    
    # Precompute prime factorization cache
    pf_cache = {}
    
    def get_pf(n):
        if n in pf_cache:
            return pf_cache[n]
        if n == 1:
            pf = (0, 0, 0, 0, 0)
        else:
            pf_list = [0, 0, 0, 0, 0]
            temp = n
            for i, p in enumerate(primes):
                while temp % p == 0:
                    pf_list[i] += 1
                    temp //= p
            if temp != 1:
                pf = None
            else:
                pf = tuple(pf_list)
        pf_cache[n] = pf
        return pf
    
    # Generate all divisors of 12! greater than 12
    divisors = [1]
    for i, p in enumerate(primes):
        new_divs = []
        for d in divisors:
            for e in range(max_exp[i] + 1):
                new_divs.append(d * (p ** e))
        divisors = new_divs
    
    valid_divs = sorted([d for d in divisors if d > 12])
    
    # Map each valid divisor to its pf
    div_pf = {}
    for d in valid_divs:
        div_pf[d] = get_pf(d)
    
    # Precompute prefix products for min_product pruning
    # For efficiency, we'll compute on the fly but with better bounds
    
    best_max_val = N
    best_factors = [N]
    
    def dfs(last_factor, rem_pf, count_left, path, current_max):
        nonlocal best_max_val, best_factors
        
        if time.time() - start_time > 85:
            return
        
        if count_left == 0:
            if rem_pf == (0, 0, 0, 0, 0):
                if current_max < best_max_val:
                    best_max_val = current_max
                    best_factors = list(path)
            return
        
        if current_max >= best_max_val:
            return
        
        # Compute remaining target value
        rem_target = 1
        for i in range(5):
            rem_target *= primes[i] ** rem_pf[i]
        
        # If only one factor left
        if count_left == 1:
            x = rem_target
            if x > last_factor and x > 12 and x < best_max_val:
                pf_x = get_pf(x)
                if pf_x is not None and pf_x == rem_pf:
                    path.append(x)
                    if x < best_max_val:
                        best_max_val = x
                        best_factors = list(path)
                    path.pop()
            return
        
        # Upper bound for next factor x:
        # x * (x+1) * (x+2) * ... * (x + count_left - 1) <= rem_target
        # This gives us x^count_left <= rem_target approximately
        # More precisely, x must satisfy that product <= rem_target
        # Also x must be > last_factor and x < best_max_val
        
        # Find upper bound: x such that x*(x+1)*...*(x+count_left-1) <= rem_target
        # Try binary search or just use geometric mean + margin
        x_upper = min(best_max_val - 1, rem_target)
        
        # Better upper bound: x^count_left <= rem_target, so x <= rem_target^(1/count_left)
        # But we need x*(x+1)*...*(x+count_left-1) <= rem_target
        # Let's compute a tighter bound
        def max_x_for_product(rem_target, count_left, last_factor):
            """Find max x > last_factor such that x*(x+1)*...*(x+count_left-1) <= rem_target"""
            lo = last_factor + 1
            hi = int(rem_target ** (1.0 / count_left)) + count_left + 10
            hi = min(hi, best_max_val)
            
            if hi <= lo:
                return lo - 1
            
            # Binary search for the largest x
            result = lo - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                # Compute mid*(mid+1)*...*(mid+count_left-1)
                prod = 1
                for j in range(count_left):
                    prod *= (mid + j)
                    if prod > rem_target:
                        break
                if prod <= rem_target:
                    result = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            return result
        
        x_upper = max_x_for_product(rem_target, count_left, last_factor)
        
        if x_upper <= last_factor:
            return
        
        # Also need: the product of the (count_left) smallest divisors > last_factor must be <= rem_target
        # And the product of the (count_left) largest divisors <= x_upper must be >= rem_target
        
        # Iterate through valid divisors in range (last_factor, x_upper]
        idx_start = bisect.bisect_right(valid_divs, last_factor)
        idx_end = bisect.bisect_right(valid_divs, x_upper)
        
        # Additional pruning: check if enough divisors exist
        if idx_end - idx_start < count_left:
            return
        
        # For each candidate x, check if the remaining can be formed
        for idx in range(idx_start, idx_end):
            if time.time() - start_time > 85:
                return
            
            x = valid_divs[idx]
            if x >= best_max_val:
                break
            
            pf_x = div_pf[x]
            if pf_x is None:
                continue
            
            # Check pf_x <= rem_pf
            can_divide = True
            for i in range(5):
                if pf_x[i] > rem_pf[i]:
                    can_divide = False
                    break
            if not can_divide:
                continue
            
            new_rem_pf = tuple(rem_pf[i] - pf_x[i] for i in range(5))
            new_count = count_left - 1
            new_rem_target = rem_target // x
            
            # Check if new_rem_target can be formed by new_count distinct factors > x
            # Min possible product: (x+1)*(x+2)*...*(x+new_count)
            min_possible = 1
            for j in range(1, new_count + 1):
                min_possible *= (x + j)
            
            if new_rem_target < min_possible:
                continue
            
            # Max possible product with factors < best_max_val
            # The largest factors we can use are (best_max_val-1), (best_max_val-2), ...
            # But they must be distinct and > x
            # For pruning, check if new_rem_target <= (best_max_val-1)*(best_max_val-2)*...*(best_max_val-new_count)
            # This is a loose upper bound
            
            path.append(x)
            dfs(x, new_rem_pf, new_count, path, max(current_max, x))
            path.pop()
    
    # Try different k values, starting from those likely to give good solutions
    # k=6,7,8 are most promising based on geometric mean analysis
    for k in [6, 7, 8, 5, 9, 4, 10, 3, 11, 2, 1]:
        if time.time() - start_time > 82:
            break
        dfs(12, target_pf, k, [], 12)
        if best_max_val <= 22:
            break
    
    # If we found a good solution, return it
    if best_factors != [N]:
        return best_factors
    
    # Fallback: try a few known decompositions
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    # Try to construct manually
    test_decomps = [
        [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],  # product way too big
        [14, 15, 16, 18, 20, 22, 24, 26, 28],
        [15, 16, 18, 20, 22, 24, 28, 30],
        [16, 18, 20, 24, 28, 30, 36],
    ]
    
    for decomp in test_decomps:
        prod = 1
        for v in decomp:
            prod *= v
        if prod == N and all(v > 12 for v in decomp) and len(set(decomp)) == len(decomp):
            return decomp
    
    # Last resort: return [N]
    return [N]
