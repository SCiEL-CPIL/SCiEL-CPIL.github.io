import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin

    def solve_k(k):
        """
        Find k distinct integers d1 < ... < dk >= 5 such that sum(1/di) = 1.
        Returns the list of denominators or None.
        """
        
        def dfs(start, remaining, target, current):
            if time.time() > deadline:
                return None
            
            if remaining == 0:
                if target == 0:
                    return current
                return None
            
            if target <= 0:
                return None
            
            # Pruning 1: Max possible sum with 'remaining' terms starting from 'start'
            # is sum(1/(start + i) for i in range(remaining))
            # If this is less than target, we can't reach target.
            max_possible = sum(Fraction(1, start + i) for i in range(remaining))
            if max_possible < target:
                return None
            
            # Pruning 2: Min possible sum with 'remaining' terms starting from 'start'
            # The terms must be distinct and >= start.
            # The smallest sum is achieved by taking the largest possible denominators.
            # However, denominators are bounded by 10^7.
            # A simpler lower bound: The sum of 'remaining' terms is at least 
            # 'remaining' / (start + remaining - 1) is not quite right, 
            # actually the smallest sum is when denominators are as large as possible.
            # But we also have the constraint that each 1/d < target.
            
            # Let's determine the range for the next denominator d.
            # 1. d >= start
            # 2. 1/d <= target  => d >= ceil(1/target)
            min_d = max(start, math.ceil(1 / float(target)))
            
            # 3. We need to be able to form 'target - 1/d' with 'remaining - 1' terms starting from d+1.
            # The maximum sum of 'remaining - 1' terms starting from d+1 is 
            # sum(1/(d+1 + i) for i in range(remaining - 1)).
            # We need target - 1/d <= this max sum.
            # This implies 1/d >= target - max_sum_next.
            # This gives an upper bound on d? No, it gives a lower bound on 1/d, so an upper bound on d?
            # Wait. If target - 1/d is very small, we need very large denominators for the rest.
            # If target - 1/d is very large, we need small denominators for the rest.
            
            # Let's use the bound:
            # The smallest sum of (remaining-1) terms starting from d+1 is approx (remaining-1)/(d+remaining).
            # We need target - 1/d >= (remaining-1)/(d+remaining).
            # Let f(d) = target - 1/d - (remaining-1)/(d+remaining).
            # We need f(d) >= 0.
            # As d increases, 1/d decreases, so -1/d increases (less negative).
            # (remaining-1)/(d+remaining) decreases. So -(remaining-1)/(d+remaining) increases.
            # So f(d) is increasing with d?
            # Let's check derivative.
            # d/dd (target - 1/d - C/(d+R)) = 1/d^2 - C * (-1)/(d+R)^2 = 1/d^2 + C/(d+R)^2 > 0.
            # So f(d) is increasing.
            # We need f(d) >= 0. So d must be >= some d_min_f.
            # This is another lower bound.
            
            # What about the upper bound on d?
            # The maximum sum of (remaining-1) terms starting from d+1 is 
            # sum(1/(d+1 + i) for i in range(remaining-1)).
            # We need target - 1/d <= max_sum_next(d).
            # Let g(d) = target - 1/d - max_sum_next(d).
            # We need g(d) <= 0.
            # max_sum_next(d) decreases as d increases.
            # -1/d increases as d increases.
            # So g(d) increases as d increases.
            # We need g(d) <= 0. So d must be <= some d_max_g.
            
            # Let's compute min_d and max_d more carefully.
            
            # Lower bound from 1/d <= target:
            min_d = max(min_d, start)
            
            # Lower bound from feasibility of remaining sum:
            # We need target - 1/d >= min_sum_next(d).
            # min_sum_next(d) is the sum of the smallest (remaining-1) reciprocals starting from d+1?
            # No, the smallest sum is with the LARGEST denominators.
            # But we don't have an upper bound on denominators other than 10^7.
            # However, if we pick d too large, 1/d becomes small, and we have to make up the rest with even larger denominators, making the sum even smaller.
            # Actually, the constraint is that the REMAINING sum must be achievable.
            # The MAXIMUM sum achievable with (remaining-1) terms starting from d+1 is 
            # S_max(d) = sum(1/(d+1+i) for i in range(remaining-1)).
            # We MUST have target - 1/d <= S_max(d).
            # If target - 1/d > S_max(d), it's impossible.
            # Since S_max(d) decreases as d increases, and target - 1/d increases as d increases,
            # the condition target - 1/d <= S_max(d) will hold for small d and fail for large d.
            # So this gives an UPPER bound on d.
            
            # Let's find max_d such that target - 1/d <= sum(1/(d+1+i) for i in range(remaining-1)).
            # We can estimate this.
            # For large d, 1/d is small.
            # Let's just iterate d from min_d upwards and break when the condition fails.
            
            # However, iterating up to 10^7 is too slow.
            # We can approximate S_max(d) ~ (remaining-1)/(d+1).
            # target - 1/d <= (remaining-1)/(d+1)
            # target(d+1) - (d+1)/d <= (remaining-1)
            # target*d + target - 1 - 1/d <= remaining - 1
            # target*d - 1/d <= remaining - 1 - target
            # For large d, target*d <= remaining - 1 - target
            # d <= (remaining - 1 - target) / target
            # This is a loose upper bound.
            
            # Let's use a tighter bound.
            # The sum of the first (remaining-1) terms starting from d+1 is bounded by 
            # (remaining-1)/(d+1).
            # So we need target - 1/d <= (remaining-1)/(d+1).
            # This gives an upper bound on d.
            
            # Let's calculate max_d using this approximation, then verify.
            # target - 1/d <= (remaining-1)/(d+1)
            # target(d+1) - (d+1)/d <= remaining-1
            # target*d + target - 1 - 1/d <= remaining-1
            # target*d - 1/d <= remaining - 1 - target
            # Let's just use a binary search or a simple loop to find the upper bound if needed, 
            # but for small k (like 10), the range of d is likely small.
            
            # Let's just loop d from min_d to a reasonable limit.
            # The limit can be determined by the fact that 1/d must be at least target/remaining (on average).
            # So d <= remaining / target.
            max_d = int(remaining / float(target))
            max_d = min(max_d, 10000000)
            
            if min_d > max_d:
                return None
            
            for d in range(min_d, max_d + 1):
                if time.time() > deadline:
                    return None
                
                new_target = target - Fraction(1, d)
                if new_target < 0:
                    break
                
                # Check if new_target is achievable with remaining-1 terms starting from d+1.
                # Max sum of next terms:
                if remaining - 1 > 0:
                    # Sum of 1/(d+1) ... 1/(d+remaining-1)
                    # If this sum < new_target, then we can't reach new_target.
                    # But wait, if we pick smaller denominators later, the sum is larger.
                    # The MAX sum is with the smallest denominators (d+1, d+2, ...).
                    # So if max_sum < new_target, it's impossible.
                    max_next_sum = sum(Fraction(1, d + 1 + i) for i in range(remaining - 1))
                    if max_next_sum < new_target:
                        # Since max_next_sum decreases as d increases, and new_target increases as d increases,
                        # if this fails for this d, it will fail for all larger d.
                        break
                    
                    # Min sum of next terms:
                    # The min sum is with the largest denominators.
                    # But we don't have a strict upper bound on denominators other than 10^7.
                    # However, we can check if new_target is too small.
                    # The smallest sum of (remaining-1) terms starting from d+1 is 
                    # achieved by taking the largest possible denominators.
                    # But we can always make the sum arbitrarily small by taking large denominators.
                    # So the only constraint is that new_target > 0.
                    # Wait, we also need distinct integers.
                    # The smallest sum is not 0, it's bounded by the fact that denominators are integers.
                    # But for our purpose, if new_target > 0, it's potentially achievable.
                    # However, we must ensure that the denominators don't exceed 10^7.
                    # If new_target is very small, we might need very large denominators.
                    # 1/d_min >= new_target / (remaining-1) => d_min <= (remaining-1)/new_target.
                    # If (remaining-1)/new_target > 10^7, it might be impossible?
                    # No, we can always pick large denominators.
                    # But if we need the sum to be EXACTLY new_target, we have freedom.
                    # The only issue is if new_target is so small that we can't represent it 
                    # with (remaining-1) distinct unit fractions with denominators <= 10^7.
                    # But for k=10, new_target is not that small.
                    
                res = dfs(d + 1, remaining - 1, new_target, current + [d])
                if res is not None:
                    return res
            
            return None
        
        return dfs(5, k, Fraction(1, 1), [])

    # Try to find a solution with k=10
    result = solve_k(10)
    if result:
        return result

    # If k=10 fails, try k=11, etc.
    # But k=10 is the minimum, so if it fails, we return empty or a valid larger set.
    # For now, return empty if not found (should not happen).
    return []

# EVOLVE-BLOCK-END
