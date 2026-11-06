import time
import math
from fractions import Fraction
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    def find_solution(k):
        """
        Find k distinct denominators d1 < ... < dk in [5, 10^7] 
        such that sum(1/di) = 1.
        Returns list of denominators or None.
        """
        target = Fraction(1, 1)
        max_den = 10_000_000
        
        def dfs(rem, min_den, current, depth):
            if time.time() > deadline:
                return None
            
            # Base case
            if depth == k:
                if rem == 0:
                    return current[:]
                return None
            
            # Pruning:
            # 1. rem must be positive
            if rem <= 0:
                return None
            
            # 2. Max possible sum of remaining (k - depth) terms
            # The smallest denominators we can pick are min_den, min_den+1, ...
            # So max sum is sum(1/(min_den + i) for i in range(k-depth))
            # If rem > max_sum, impossible.
            max_sum = Fraction(0, 1)
            for i in range(k - depth):
                max_sum += Fraction(1, min_den + i)
            if rem > max_sum:
                return None
            
            # 3. Min possible sum of remaining (k - depth) terms
            # The largest denominators are max_den, max_den-1, ...
            # But they must be distinct and > last picked.
            # Approx min sum is (k-depth) / max_den.
            # If rem < min_sum, impossible.
            min_sum = Fraction(k - depth, max_den)
            if rem < min_sum:
                return None
            
            # 4. Bounds for next denominator d
            # 1/d <= rem  => d >= 1/rem
            # 1/d >= rem / (k - depth) ? No.
            # We need 1/d + max_possible_rest >= rem.
            # max_possible_rest for k-depth-1 terms starting from d+1 is sum(1/(d+1+i)).
            # So 1/d + sum(1/(d+1+i)) >= rem.
            
            # Lower bound for d:
            # d >= ceil(1/rem)
            d_min = max(min_den, math.ceil(1 / float(rem)))
            if d_min < min_den:
                d_min = min_den
            
            # Upper bound for d:
            # 1/d <= rem => d >= 1/rem (already handled)
            # Also 1/d >= rem - max_possible_rest.
            # This is tricky. 
            # Simple upper bound: d <= max_den.
            # Also, we need enough room for the remaining terms.
            # The smallest possible value for the remaining k-depth-1 terms is roughly (k-depth-1)/max_den.
            # So 1/d <= rem - (k-depth-1)/max_den.
            # d >= 1 / (rem - (k-depth-1)/max_den).
            
            # Let's iterate d from d_min upwards.
            # To optimize, we can also bound d by:
            # 1/d <= rem => d >= 1/rem
            # And we need to be able to form the remainder with k-depth-1 terms.
            # The maximum sum of k-depth-1 terms with denominators > d is sum(1/(d+1+i)).
            # If rem - 1/d > sum(1/(d+1+i)), then d is too small (we need a larger 1/d? No, smaller d means larger 1/d).
            # Wait. If d is small, 1/d is large.
            # If 1/d > rem, we skip (handled by d_min).
            # If 1/d < rem - max_possible_rest, then even if we pick the smallest possible next denominators, we can't reach rem.
            # So we need 1/d >= rem - max_possible_rest.
            # d <= 1 / (rem - max_possible_rest).
            
            # Calculate d_max
            # max_possible_rest for k-depth-1 terms starting from d+1.
            # This depends on d. So we can't easily compute a static d_max.
            # However, as d increases, 1/d decreases, and max_possible_rest decreases.
            # The condition 1/d + max_possible_rest >= rem becomes harder to satisfy as d increases?
            # 1/d decreases. max_possible_rest decreases (denominators get larger).
            # So LHS decreases.
            # So if condition fails for some d, it fails for all larger d.
            # Thus, we can break the loop when the condition fails.
            
            d = d_min
            while d <= max_den:
                # Check time
                if time.time() > deadline:
                    return None
                
                # Check if 1/d is too big
                if Fraction(1, d) > rem:
                    break # d is too small? No, 1/d > rem means d < 1/rem. 
                    # But we started from d_min = ceil(1/rem). So 1/d <= rem.
                    # Actually if rem is small, 1/rem is large.
                    # If d_min is calculated correctly, 1/d <= rem.
                    
                # Calculate remainder
                new_rem = rem - Fraction(1, d)
                
                # Check if we can form new_rem with k-depth-1 terms > d
                # Max sum of k-depth-1 terms starting from d+1
                max_rest = Fraction(0, 1)
                for i in range(k - depth - 1):
                    max_rest += Fraction(1, d + 1 + i)
                    
                if new_rem > max_rest:
                    # We need a larger 1/d (smaller d) to reduce new_rem?
                    # No, if new_rem > max_rest, it means we subtracted too little.
                    # We need 1/d to be LARGER, so d must be SMALLER.
                    # But we are increasing d. So if it's too small now, it will be even smaller later.
                    # Wait.
                    # new_rem = rem - 1/d.
                    # If d increases, 1/d decreases, new_rem INCREASES.
                    # max_rest DECREASES.
                    # So if new_rem > max_rest, it will get worse for larger d.
                    # So we should break?
                    # But we started from the smallest d.
                    # If the smallest d gives new_rem > max_rest, then no d works?
                    # Let's check.
                    # We need rem - 1/d <= max_rest(d).
                    # Let f(d) = rem - 1/d - max_rest(d).
                    # We need f(d) <= 0.
                    # As d increases, 1/d decreases -> -1/d increases.
                    # max_rest(d) decreases -> -max_rest(d) increases.
                    # So f(d) is increasing.
                    # If f(d_min) > 0, then f(d) > 0 for all d >= d_min.
                    # So we should return None (or break).
                    return None
                
                # Min sum of k-depth-1 terms
                # They must be > d and <= max_den.
                # Min sum is approx (k-depth-1) / max_den.
                # If new_rem < min_rest, then 1/d was too big (d too small).
                # We need 1/d smaller, so d larger.
                # So we continue to next d.
                min_rest = Fraction(k - depth - 1, max_den)
                if new_rem < min_rest:
                    continue
                
                # Recursive step
                result = dfs(new_rem, d + 1, current + [d], depth + 1)
                if result:
                    return result
                
                d += 1
                
            return None

        return dfs(target, 5, [], 0)

    # Try k from 4 upwards
    for k in range(4, 11):
        sol = find_solution(k)
        if sol:
            # Verify
            s = Fraction(0, 1)
            for d in sol:
                s += Fraction(1, d)
            if s == 1 and len(set(sol)) == k and all(5 <= d <= 10_000_000 for d in sol):
                return sol
            else:
                # Fallback if verification fails (should not happen)
                continue
        
        if time.time() > deadline - 5.0:
            break

    # Fallback: Return a known valid solution if found, or raise error
    # A known solution for k=6: 
    # 1 = 1/5 + 1/6 + 1/7 + 1/42 + 1/180 + 1/... ?
    # Let's just return the first found valid one.
    # If none found, we must return something valid.
    # The problem implies a solution exists.
    # For safety, return a hardcoded valid solution if search fails.
    # One valid solution is [5, 25, 31, 3125] ... wait, I disproved that earlier.
    # Let's use a verified 6-term solution.
    # 1/5 + 1/6 + 1/7 + 1/42 = 8/15. Rem 7/15.
    # 7/15 = 1/15 + 1/15? No distinct.
    # 7/15 = 1/10 + 1/6? 1/10+1/6 = 3/30+5/30=8/30=4/15. No.
    # 7/15 = 1/12 + 1/5? 1/12+1/5 = 5/60+12/60=17/60. No.
    
    # I will rely on the search finding k=5 or k=6.
    # If it times out, I'll return a dummy list that might fail, but better to return a valid one.
    # Let's try to construct one manually for fallback.
    # 1/5 + 1/6 + 1/7 + 1/42 + 1/180 + 1/...
    # 8/15 + 1/180 = 96/180 + 1/180 = 97/180.
    # Rem = 1 - 97/180 = 83/180.
    # 83/180 is prime numerator? 83 is prime.
    # 1/x + 1/y = 83/180.
    # x,y must be large.
    
    # Actually, the search should find it.
    raise RuntimeError("No solution found")
