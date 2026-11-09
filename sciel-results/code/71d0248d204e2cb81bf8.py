import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    def dfs_exact(rem_sum, min_d, remaining_terms, current_list):
        if time.time() > deadline:
            return None
            
        if remaining_terms == 0:
            if rem_sum == 0:
                return current_list
            return None
            
        if remaining_terms == 1:
            # rem_sum must be 1/d for some integer d >= min_d
            # 1/d = rem_sum => d = 1/rem_sum
            if rem_sum > 0:
                # Check if rem_sum is a unit fraction 1/d
                # rem_sum = Fraction(num, den). d = den/num.
                # d must be integer, so num must divide den.
                if rem_sum.numerator == 1:
                    d = rem_sum.denominator
                    if d >= min_d and d >= 5 and d <= 10000000:
                        return current_list + [d]
            return None

        # Pruning: Max possible sum with remaining_terms terms starting from min_d
        # The smallest denominators are min_d, min_d+1, ...
        # If the sum of reciprocals of the smallest 'remaining_terms' denominators
        # is less than rem_sum, it's impossible.
        
        max_possible_sum = Fraction(0, 1)
        for i in range(remaining_terms):
            max_possible_sum += Fraction(1, min_d + i)
            
        if max_possible_sum < rem_sum:
            return None

        # Lower bound for d:
        # We need 1/d <= rem_sum => d >= ceil(1/rem_sum)
        # Also d >= min_d
        if rem_sum > 0:
            # d >= 1/rem_sum
            d_min_candidate = int(1 / rem_sum) + 1
            if Fraction(1, d_min_candidate) > rem_sum:
                 d_min_candidate += 1 # Ensure 1/d <= rem_sum
            # Actually simpler: d >= ceil(1/rem_sum)
            # 1/rem_sum might be integer. If rem_sum = 1/10, 1/rem = 10. d can be 10.
            # So d >= ceil(1/rem_sum).
            inv_rem = 1 / rem_sum
            d_min = max(min_d, int(inv_rem) + (1 if inv_rem != int(inv_rem) else 0))
        else:
            return None

        # Upper bound for d:
        # We have 'remaining_terms' terms. The first is d.
        # The sum of the remaining 'remaining_terms - 1' terms must be at least rem_sum - 1/d.
        # The maximum sum of the remaining terms is bounded by (remaining_terms - 1) / (d+1).
        # So we need rem_sum - 1/d <= (remaining_terms - 1) / (d+1)
        # This is a tight bound. A looser but easier bound is:
        # rem_sum <= remaining_terms / d  => d <= remaining_terms / rem_sum
        d_max = int(remaining_terms / rem_sum)
        if d_max < d_min:
            return None
        
        # Cap d_max to avoid huge loops if rem_sum is very small
        if d_max > 10000000:
            d_max = 10000000

        # Iterate d from d_min to d_max
        for d in range(d_min, d_max + 1):
            if time.time() > deadline:
                return None
                
            # Check if 1/d is too big
            if Fraction(1, d) > rem_sum:
                break
                
            rem_new = rem_sum - Fraction(1, d)
            
            # Prune: Check if it's possible to fill rem_new with remaining_terms - 1 terms >= d+1
            # Max sum of remaining terms: sum(1/(d+1+i) for i in range(remaining_terms-1))
            if remaining_terms > 1:
                max_next_sum = Fraction(0, 1)
                for i in range(remaining_terms - 1):
                    max_next_sum += Fraction(1, d + 1 + i)
                if rem_new > max_next_sum:
                    # Since d is increasing, 1/d is decreasing, rem_new is increasing.
                    # Wait, if d increases, 1/d decreases, so rem_new = rem_sum - 1/d increases.
                    # The available denominators also increase (d+1...), so their reciprocal sum decreases.
                    # So if rem_new > max_next_sum, it will get worse for larger d?
                    # Let's check:
                    # rem_new(d) = R - 1/d. Increases with d.
                    # max_next(d) = sum(1/(d+1+i)). Decreases with d.
                    # So if rem_new(d) > max_next(d), then for d' > d, rem_new(d') > rem_new(d) and max_next(d') < max_next(d).
                    # So the inequality will persist. We can break.
                    break
                    
            res = dfs_exact(rem_new, d + 1, remaining_terms - 1, current_list + [d])
            if res is not None:
                return res
                
        return None

    # Try increasing k starting from 10
    # Based on analysis, k=9 is impossible (max sum < 1). k=10 is the minimum candidate.
    for k in range(10, 16):
        res = dfs_exact(Fraction(1, 1), 5, k, [])
        if res is not None:
            # Validate
            s = Fraction(0, 1)
            for d in res:
                s += Fraction(1, d)
            if s == 1 and len(set(res)) == len(res) and all(5 <= d <= 10000000 for d in res):
                return res

    # Fallback: Use the subset DP approach which is guaranteed to find a solution
    # This solution will likely have more terms, but is valid.
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
            
    # If even DP fails (should not happen for 2520), try a larger target or raise
    # 2520 is divisible by 5,6,7,8,9,10,12,14,15,18,20,21,24,28,30,35,36,40,42,45,56,60,63,70,72,84,90,105,120,126,140,168,180,210,252,280,315,360,420,504,630,840,1260,2520
    # It is very likely to find a combination.
    
    raise RuntimeError("no solution found")
