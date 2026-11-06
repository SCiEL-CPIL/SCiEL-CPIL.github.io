import time
import random

def construct():
    deadline = time.time() + 85.0
    n = 512
    target_size = 48
    
    def _build_dp(arr):
        m = len(arr)
        if m == 0:
            return [1]
        dp = [0] * (m + 1)
        dp[0] = 1
        for val in arr:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << val)
        return dp

    def _is_valid_incremental(existing, x, dp_cache, sum_cache):
        m = len(existing)
        total_sum = sum_cache
        
        for k in range(2, m + 1):
            target = k * x
            if target > total_sum:
                break
            if (dp_cache[k] >> target) & 1:
                return False
        
        for a in existing:
            for t in range(1, m + 1):
                target = (t + 1) * a - x
                if target < 0:
                    continue
                if target > total_sum:
                    break
                if (dp_cache[t] >> target) & 1:
                    return False
        return True

    def _greedy_from_order(order, start_set=None):
        if start_set is None:
            selected = []
        else:
            selected = list(start_set)
        
        dp_cache = _build_dp(selected)
        sum_cache = sum(selected)
        
        for x in order:
            if len(selected) >= target_size:
                break
            if x in selected:
                continue
            if _is_valid_incremental(selected, x, dp_cache, sum_cache):
                selected.append(x)
                # Update DP
                m = len(selected) - 1
                new_dp = [0] * (len(selected) + 1)
                new_dp[0] = 1
                for k in range(1, m + 1):
                    new_dp[k] = dp_cache[k] | (dp_cache[k-1] << x)
                new_dp[m+1] = dp_cache[m] << x
                dp_cache = new_dp
                sum_cache += x
                
        return selected

    best = []
    
    # Strategy 1: Start with powers of 2
    powers = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    order1 = powers + [x for x in range(1, n+1) if x not in powers]
    res1 = _greedy_from_order(order1, powers)
    if len(res1) > len(best): best = res1

    # Strategy 2: Greedy from natural order
    order2 = list(range(1, n+1))
    res2 = _greedy_from_order(order2)
    if len(res2) > len(best): best = res2

    # Strategy 3: Greedy from descending order
    order3 = list(range(n, 0, -1))
    res3 = _greedy_from_order(order3)
    if len(res3) > len(best): best = res3

    # Strategy 4: Randomized greedy
    for trial in range(200):
        if time.time() > deadline - 10.0:
            break
        random.seed(trial)
        order = list(range(1, n+1))
        random.shuffle(order)
        res = _greedy_from_order(order)
        if len(res) > len(best):
            best = res
            if len(best) >= target_size:
                return best

    return best
