import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort()  # ascending
    m = len(divisors)  # 60 divisors
    
    # For each cardinality k from 1 upward, check if any k distinct divisors sum to target
    # Use combinations for small k, switch to smarter method for larger k
    
    for k in range(1, m + 1):
        if time.time() > deadline:
            break
        
        if k == 1:
            if target in divisors:
                return [target]
        elif k == 2:
            # Two-sum: for each d, check if target-d is in divisors and > d
            div_set = set(divisors)
            for i in range(m):
                d1 = divisors[i]
                d2 = target - d1
                if d2 > d1 and d2 in div_set:
                    return [d1, d2]
        elif k == 3:
            # Three-sum
            div_set = set(divisors)
            for i in range(m):
                d1 = divisors[i]
                rem2 = target - d1
                for j in range(i + 1, m):
                    d2 = divisors[j]
                    d3 = rem2 - d2
                    if d3 > d2 and d3 in div_set:
                        return [d1, d2, d3]
        elif k == 4:
            # Four-sum
            div_set = set(divisors)
            for i in range(m):
                d1 = divisors[i]
                rem3 = target - d1
                for j in range(i + 1, m):
                    d2 = divisors[j]
                    rem2 = rem3 - d2
                    for l in range(j + 1, m):
                        d3 = divisors[l]
                        d4 = rem2 - d3
                        if d4 > d3 and d4 in div_set:
                            return [d1, d2, d3, d4]
        elif k == 5:
            # Five-sum: use two-sum lookup for the last two
            div_set = set(divisors)
            # Precompute all two-sums with their pairs
            two_sums = {}
            for i in range(m):
                for j in range(i + 1, m):
                    s = divisors[i] + divisors[j]
                    if s not in two_sums:
                        two_sums[s] = (divisors[i], divisors[j])
            
            for i in range(m):
                d1 = divisors[i]
                rem4 = target - d1
                for j in range(i + 1, m):
                    d2 = divisors[j]
                    rem3 = rem4 - d2
                    for l in range(j + 1, m):
                        d3 = divisors[l]
                        rem2 = rem3 - d3
                        if rem2 in two_sums:
                            d4, d5 = two_sums[rem2]
                            if d4 > d3 and d5 > d4:
                                return [d1, d2, d3, d4, d5]
        else:
            # For k >= 6, use DP: dp[s] = min count to achieve sum s
            # This is the original approach but only run if needed
            INF = float('inf')
            dp = [INF] * (target + 1)
            dp[0] = 0
            # Track which divisors were used via a bitset per sum (too large)
            # Instead, just find min count, then backtrack
            # Use a different DP: for each divisor, update
            # But we need to ensure distinctness. Use 0/1 knapsack.
            
            # Actually, for k >= 6, let's use a smarter meet-in-the-middle
            # Split divisors into two halves of 30 each
            half = m // 2
            left = divisors[:half]
            right = divisors[half:]
            
            # For left half, compute all (sum, count) pairs
            # 2^30 is too many, so we need pruning
            # Instead, use DP on the left half to get min count per sum
            dp_left = [INF] * (target + 1)
            dp_left[0] = 0
            for d in left:
                if d > target:
                    continue
                for s in range(target, d - 1, -1):
                    if dp_left[s - d] + 1 < dp_left[s]:
                        dp_left[s] = dp_left[s - d] + 1
            
            dp_right = [INF] * (target + 1)
            dp_right[0] = 0
            for d in right:
                if d > target:
                    continue
                for s in range(target, d - 1, -1):
                    if dp_right[s - d] + 1 < dp_right[s]:
                        dp_right[s] = dp_right[s - d] + 1
            
            # Combine: find min dp_left[s] + dp_right[target-s]
            best_count = INF
            best_s = -1
            for s in range(target + 1):
                if dp_left[s] != INF and dp_right[target - s] != INF:
                    total = dp_left[s] + dp_right[target - s]
                    if total < best_count:
                        best_count = total
                        best_s = s
            
            if best_count != INF and best_count >= k:
                # Backtrack to find actual divisors
                # Reconstruct from dp_left and dp_right
                selected_left = []
                s = best_s
                # Re-run DP on left with parent tracking
                dp_l = [INF] * (target + 1)
                dp_l[0] = 0
                parent_l = [-1] * (target + 1)
                for d in left:
                    if d > target:
                        continue
                    for s2 in range(target, d - 1, -1):
                        if dp_l[s2 - d] + 1 < dp_l[s2]:
                            dp_l[s2] = dp_l[s2 - d] + 1
                            parent_l[s2] = d
                
                s = best_s
                while s > 0 and parent_l[s] != -1:
                    selected_left.append(parent_l[s])
                    s -= parent_l[s]
                
                dp_r = [INF] * (target + 1)
                dp_r[0] = 0
                parent_r = [-1] * (target + 1)
                for d in right:
                    if d > target:
                        continue
                    for s2 in range(target, d - 1, -1):
                        if dp_r[s2 - d] + 1 < dp_r[s2]:
                            dp_r[s2] = dp_r[s2 - d] + 1
                            parent_r[s2] = d
                
                s = target - best_s
                selected_right = []
                while s > 0 and parent_r[s] != -1:
                    selected_right.append(parent_r[s])
                    s -= parent_r[s]
                
                result = selected_left + selected_right
                if len(set(result)) == len(result) and sum(result) == target:
                    return result
        
        # If we get here, no solution found for this k, continue to next k
    
    # Final fallback: greedy
    divisors_desc = sorted(divisors, reverse=True)
    remainder = target
    chosen = []
    for d in divisors_desc:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    return chosen
