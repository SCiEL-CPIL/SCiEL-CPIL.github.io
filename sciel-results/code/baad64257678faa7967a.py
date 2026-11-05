import time
import sys
from fractions import Fraction
from math import gcd

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s limit
    
    def elapsed():
        return time.time() - start_time
    
    def time_ok():
        return elapsed() < deadline
    
    # We want the minimum number of distinct denominators in [5, 10000000]
    # such that their reciprocal sum is exactly 1.
    
    # k=1: 1/5 != 1. Impossible.
    # k=2: 1/a + 1/b = 1 => (a-1)(b-1)=1 => a=2, b=2. But d>=5. Impossible.
    # k=3: 1/a + 1/b + 1/c = 1.
    #   Max sum with d>=5 is 1/5+1/6+1/7 = 41/105 < 1. Impossible.
    # k=4: 1/a + 1/b + 1/c + 1/d = 1.
    #   Max sum with d>=5 is 1/5+1/6+1/7+1/8 = 205/840 = 41/168 < 1. Impossible.
    # k=5: 1/a + 1/b + 1/c + 1/d + 1/e = 1.
    #   Max sum with d>=5 is 1/5+1/6+1/7+1/8+1/9 = 1429/2520 < 1. Impossible.
    #   Wait, 1/5+1/6+1/7+1/8+1/9 = (504+420+360+315+280)/2520 = 1879/2520 < 1.
    #   So k=5 is impossible?
    #   Let's check k=6.
    #   Max sum with d>=5 is 1/5+1/6+1/7+1/8+1/9+1/10 = 1879/2520 + 252/2520 = 2131/2520 < 1.
    #   Still < 1.
    #   k=7: 1/5..1/11. Sum = 2131/2520 + 1/11 = (2131*11 + 2520) / 27720 = (23441 + 2520)/27720 = 25961/27720 < 1.
    #   k=8: 1/5..1/12. Sum = 25961/27720 + 1/12 = (25961*2 + 2310)/55440 = (51922+2310)/55440 = 54232/55440 < 1.
    #   k=9: 1/5..1/13. Sum = 54232/55440 + 1/13 = (54232*13 + 55440) / 720720 = (705016 + 55440)/720720 = 760456/720720 > 1.
    #   So k=9 is possible.
    
    # Let's search for k=5, 6, 7, 8, 9 using DFS with strong pruning.
    
    def dfs(rem, min_d, depth, max_depth, path):
        if not time_ok():
            return None
        if rem == 0:
            return path
        if depth == max_depth:
            return None
        
        terms_left = max_depth - depth
        
        # Pruning 1: Upper bound on rem
        # Sum of 1/d for d from min_d to min_d+terms_left-1 is an upper bound for the remaining sum
        # If rem > that sum, impossible.
        # But computing that sum exactly is expensive.
        # Approximation: rem <= terms_left / min_d
        # If rem > terms_left / min_d, return None.
        if rem > terms_left / min_d:
            return None
        
        # Pruning 2: Lower bound on rem
        # Sum of 1/d for d from min_d to infinity is 1/min_d + 1/(min_d+1) + ...
        # This converges. But we only have terms_left terms.
        # The smallest possible sum is 1/(L) + 1/(L+1) + ... where L is the largest possible denominator.
        # Actually, the denominators must be distinct and >= min_d.
        # The smallest sum for terms_left terms with denominators >= min_d is achieved by taking the largest possible denominators.
        # But denominators are unbounded (up to 10^7).
        # However, 1/d must be >= rem / terms_left.
        # So d <= terms_left / rem.
        
        # Upper bound for next denominator d:
        # 1/d <= rem  =>  d >= 1/rem
        # Also, the sum of the remaining terms (including this one) is rem.
        # 1/d + (terms_left - 1) * (1/(d+1)) >= rem  (approx lower bound for the rest)
        # Actually, 1/d + 1/(d+1) + ... + 1/(d+terms_left-1) >= rem
        # This gives an upper bound on d.
        # Let's use: 1/d <= rem / 1  => d >= ceil(1/rem)
        # And: 1/d + (terms_left - 1) / (d+1) >= rem
        # 1/d + (terms_left - 1) / (d+1) >= rem
        # (d+1 + d(terms_left - 1)) / (d(d+1)) >= rem
        # (d + 1 + d*terms_left - d) / (d^2 + d) >= rem
        # (1 + d*terms_left) / (d^2 + d) >= rem
        # 1 + d*terms_left >= rem * d^2 + rem * d
        # rem * d^2 + (rem - terms_left) * d - 1 <= 0
        # This is a quadratic in d. We can solve for d.
        
        # Simpler upper bound:
        # 1/d + (terms_left - 1) * (1/(d+1)) >= rem
        # This is a lower bound on the sum of the remaining terms given the first is 1/d.
        # Actually, the sum is maximized when denominators are smallest.
        # So if 1/d + 1/(d+1) + ... + 1/(d+terms_left-1) < rem, then this d is too large.
        # Let's approximate: 1/d + (terms_left - 1) / (d+1) >= rem
        # This is a bit complex. Let's use a simpler bound:
        # d <= terms_left / rem
        # Because 1/d + ... + 1/d = terms_left/d >= rem => d <= terms_left/rem.
        # This is a valid upper bound.
        
        d_min = max(min_d, (1 + rem.denominator - 1) // rem.numerator)  # ceil(1/rem)
        d_max = (terms_left * rem.denominator) // rem.numerator  # floor(terms_left / rem)
        
        if d_min > d_max:
            return None
        
        # Iterate d from d_min to d_max
        for d in range(d_min, d_max + 1):
            if not time_ok():
                return None
            new_rem = rem - Fraction(1, d)
            if new_rem < 0:
                break
            res = dfs(new_rem, d + 1, depth + 1, max_depth, path + [d])
            if res is not None:
                return res
        return None
    
    # Try k=5 first, then 6, etc.
    for k in range(5, 10):
        if not time_ok():
            break
        res = dfs(Fraction(1, 1), 5, 0, k, [])
        if res is not None:
            return res
    
    # Fallback: If DFS fails, try a different approach or return a known solution if we have one.
    # But we should find one.
    
    # Let's try a more direct search for k=9 or 10 if DFS is too slow.
    # Actually, let's increase the deadline margin for DFS.
    
    # If we reach here, we haven't found a solution.
    # We must return something valid.
    # Let's try to construct a solution greedily or with a different method.
    
    # Greedy method (Fibonacci-Sylvester) often produces large denominators.
    # 1 = 1/5 + 4/5
    # 4/5 = 1/2 + 3/10 -> but 2 < 5.
    # Greedy doesn't work well with lower bound constraints.
    
    # Let's try a randomized search or a more efficient DFS.
    
    # Actually, the DFS above should be fast enough for k=9.
    # Let's add a fallback that returns a valid solution if we find one later.
    
    # For now, if DFS fails, we raise an error or return a default.
    # But the task says "Return 1..256 distinct integer denominators...".
    # So we must return a valid list.
    
    # Let's try one more time with a slightly different pruning strategy.
    
    def dfs_v2(rem, min_d, depth, max_depth, path):
        if not time_ok():
            return None
        if rem == 0:
            return path
        if depth == max_depth:
            return None
        
        terms_left = max_depth - depth
        
        # Upper bound: rem <= terms_left / min_d
        if rem > terms_left / min_d:
            return None
        
        # Lower bound: rem >= 1/(min_d) + 1/(min_d+1) + ... ? No, that's the max sum.
        # The min sum is when denominators are as large as possible.
        # But denominators are bounded by 10^7.
        # 1/d + ... + 1/d' >= rem
        # This is hard to bound from below easily.
        
        d_min = max(min_d, (1 + rem.denominator - 1) // rem.numerator)
        d_max = (terms_left * rem.denominator) // rem.numerator
        
        if d_min > d_max:
            return None
        
        # Heuristic: try larger d first? Or smaller?
        # Smaller d means larger 1/d, so rem decreases faster.
        # Let's try smaller d first.
        
        for d in range(d_min, d_max + 1):
            if not time_ok():
                return None
            new_rem = rem - Fraction(1, d)
            if new_rem < 0:
                break
            res = dfs_v2(new_rem, d + 1, depth + 1, max_depth, path + [d])
            if res is not None:
                return res
        return None
    
    for k in range(5, 10):
        if not time_ok():
            break
        res = dfs_v2(Fraction(1, 1), 5, 0, k, [])
        if res is not None:
            return res
    
    # If still no solution, we have a problem.
    # Let's return a dummy solution that is valid but not minimal?
    # No, we must minimize the count.
    # But if we can't find the minimum, we should return any valid solution.
    # Actually, the evaluator measures the raw value (count).
    # So returning a valid solution with count k is better than none.
    
    # Let's try to find ANY solution with k up to 20.
    for k in range(5, 21):
        if not time_ok():
            break
        res = dfs_v2(Fraction(1, 1), 5, 0, k, [])
        if res is not None:
            return res
    
    # If we still fail, we raise an error.
    raise RuntimeError("No solution found")
