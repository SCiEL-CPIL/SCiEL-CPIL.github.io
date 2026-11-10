import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s kill
    
    target = Fraction(5, 31)
    best_max = float('inf')
    best_result = None

    def update_best(denoms):
        nonlocal best_max, best_result
        mx = max(denoms)
        if mx < best_max:
            best_max = mx
            best_result = list(denoms)

    def dfs(rem, min_d, depth, max_depth, current):
        nonlocal best_max, best_result
        if time.time() > deadline:
            return False
        
        if rem <= 0:
            return False
            
        if depth == max_depth:
            # Check if rem is exactly 1/d
            d_frac = 1 / rem
            if d_frac.denominator == 1:
                d = d_frac.numerator
                if d >= min_d and d < best_max:
                    update_best(current + [d])
                    return True
            return False
        
        # Pruning:
        # 1. Max possible sum with remaining terms
        # rem must be <= sum of largest possible reciprocals
        # Largest reciprocals are 1/min_d, 1/(min_d+1), ...
        # But a simpler upper bound is (max_depth - depth) / min_d
        if rem > Fraction(max_depth - depth, 1) / min_d:
            return False
            
        # 2. Next denominator d must be >= 1/rem
        min_next_d = max(min_d, (1 / rem).numerator // (1 / rem).denominator + 1)
        
        # 3. Next denominator d must be < best_max
        max_next_d = best_max - 1
        
        if min_next_d > max_next_d:
            return False

        for d in range(min_next_d, max_next_d + 1):
            if time.time() > deadline:
                return False
                
            new_rem = rem - Fraction(1, d)
            if new_rem <= 0:
                continue
                
            # Pruning: Can new_rem be formed by remaining terms?
            k_rem = max_depth - depth - 1
            if k_rem > 0:
                # new_rem must be <= sum of k_rem largest possible reciprocals starting from d+1
                # Upper bound: k_rem / (d+1)
                if new_rem > Fraction(k_rem, 1) / (d + 1):
                    continue
            else:
                # k_rem == 0 is handled by depth == max_depth check above, 
                # but this loop runs for depth < max_depth.
                # So k_rem >= 1 always here.
                pass

            if dfs(new_rem, d + 1, depth + 1, max_depth, current + [d]):
                return True
                
        return False

    # Try increasing k
    for k in range(3, 15):
        if time.time() > deadline:
            break
            
        # Lower bound on max denominator for k terms summing to 5/31
        # Sum >= k / M_max => M_max >= k / (5/31) = 31k/5
        min_possible_max = (31 * k + 4) // 5
        
        if min_possible_max >= best_max:
            break
            
        found = dfs(target, 7, 0, k, [])
        if found:
            # We found a solution. We continue to see if higher k can yield smaller max denominator.
            # But due to the lower bound increasing with k, this is unlikely unless the 
            # previous solution had a very large max denominator.
            pass

    if best_result is not None:
        return best_result
        
    # Fallback
    return [7, 8, 9, 10000000]
