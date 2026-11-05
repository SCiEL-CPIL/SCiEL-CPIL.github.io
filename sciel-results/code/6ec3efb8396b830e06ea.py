import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin

    def dfs(rem_sum, last_d, depth, max_depth):
        if time.time() > deadline:
            return None
        
        terms_left = max_depth - depth
        
        if terms_left == 0:
            if rem_sum == 0:
                return []
            return None
        
        # Lower bound: 1/d <= rem_sum => d >= 1/rem_sum
        # Also d > last_d
        min_d = max(last_d + 1, int((1 / rem_sum).numerator // rem_sum.denominator) + (1 if (1 / rem_sum).numerator % rem_sum.denominator else 0))
        # Simpler: ceil(1/rem_sum)
        import math
        min_d = max(last_d + 1, math.ceil(1 / rem_sum))
        
        # Upper bound: sum_{j=0}^{terms_left-1} 1/(d+j) >= rem_sum
        # Approx d <= terms_left / rem_sum
        max_d = int(terms_left / rem_sum) + 20
        
        for d in range(min_d, max_d + 1):
            if d < 5:
                continue
            
            next_rem = rem_sum - Fraction(1, d)
            if next_rem < 0:
                break
            
            if terms_left - 1 == 0:
                if next_rem == 0:
                    return [d]
                continue
            
            # Pruning: Max sum of remaining terms
            # If max sum < next_rem, then for all larger d, max sum decreases and next_rem increases, so condition holds
            max_possible_sum = sum(Fraction(1, d + j) for j in range(1, terms_left))
            if max_possible_sum < next_rem:
                break
            
            res = dfs(next_rem, d, depth + 1, max_depth)
            if res is not None:
                return [d] + res
        
        return None

    target_sum = Fraction(1, 1)
    
    # Try to find minimal count. We know it's at least 8.
    for max_depth in range(8, 13):
        result = dfs(target_sum, 4, 0, max_depth)
        if result is not None:
            return result
            
    raise RuntimeError("No solution found")
