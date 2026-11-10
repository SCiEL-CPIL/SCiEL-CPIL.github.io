def construct():
    from math import factorial
    from itertools import product
    from functools import lru_cache
    import sys
    import bisect
    sys.setrecursionlimit(10000)

    target = factorial(12)  # 479001600

    primes = [2, 3, 5, 7, 11]
    target_exp = [10, 5, 2, 1, 1]
    target_state_vec = tuple(target_exp)

    def gen_divisors(exp_bounds, idx=0, current=1):
        if idx == len(exp_bounds):
            return [current]
        result = []
        for e in range(exp_bounds[idx] + 1):
            result.extend(gen_divisors(exp_bounds, idx + 1, current * primes[idx] ** e))
        return result

    all_divisors = gen_divisors(target_exp)
    candidates = [d for d in all_divisors if d > 12]
    candidates.sort()
    
    def exp_vector(n):
        vec = [0] * len(primes)
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)

    cand_vecs = [exp_vector(c) for c in candidates]
    unique_vals = sorted(set(candidates))

    # DP State: (e0, e1, e2, e3, e4, count)
    # count is the number of factors used.
    # We want to reach target_state_vec with count <= 12.
    
    def can_achieve(max_val):
        idx = bisect.bisect_right(candidates, max_val)
        if idx == 0:
            return False
        
        # reachable states: set of (e0, e1, e2, e3, e4, count)
        # Initial state: (0,0,0,0,0, 0)
        reachable = {(0, 0, 0, 0, 0, 0)}
        
        for i in range(idx):
            cv = cand_vecs[i]
            new_reachable = set()
            for state in reachable:
                e0, e1, e2, e3, e4, cnt = state
                ne0 = e0 + cv[0]
                ne1 = e1 + cv[1]
                ne2 = e2 + cv[2]
                ne3 = e3 + cv[3]
                ne4 = e4 + cv[4]
                ncnt = cnt + 1
                
                if ncnt > 12:
                    continue
                if ne0 > target_exp[0] or ne1 > target_exp[1] or ne2 > target_exp[2] or ne3 > target_exp[3] or ne4 > target_exp[4]:
                    continue
                
                new_state = (ne0, ne1, ne2, ne3, ne4, ncnt)
                new_reachable.add(new_state)
            
            reachable = reachable | new_reachable
            
            # Check if any state in reachable matches target vec and count <= 12
            # Since we add to reachable, we can check after update
            for state in reachable:
                if state[0] == target_exp[0] and state[1] == target_exp[1] and state[2] == target_exp[2] and state[3] == target_exp[3] and state[4] == target_exp[4] and state[5] <= 12:
                    return True
        
        return False

    # Binary search
    lo_idx = 0
    hi_idx = len(unique_vals) - 1
    best = None
    best_idx = -1
    
    while lo_idx <= hi_idx:
        mid_idx = (lo_idx + hi_idx) // 2
        mid_val = unique_vals[mid_idx]
        if can_achieve(mid_val):
            best = mid_val
            best_idx = mid_idx
            hi_idx = mid_idx - 1
        else:
            lo_idx = mid_idx + 1

    if best is None:
        return [target]

    # Reconstruct
    idx_limit = bisect.bisect_right(candidates, best)
    
    # dp[i] is a dict mapping state to (prev_state, cand_idx)
    # state is (e0, e1, e2, e3, e4, count)
    
    zero_state = (0, 0, 0, 0, 0, 0)
    
    dp = [dict() for _ in range(idx_limit + 1)]
    dp[0][zero_state] = None
    
    for i in range(idx_limit):
        # Carry over
        for state, parent in dp[i].items():
            if state not in dp[i + 1]:
                dp[i + 1][state] = parent
        
        cv = cand_vecs[i]
        for state in list(dp[i].keys()):
            e0, e1, e2, e3, e4, cnt = state
            ne0 = e0 + cv[0]
            ne1 = e1 + cv[1]
            ne2 = e2 + cv[2]
            ne3 = e3 + cv[3]
            ne4 = e4 + cv[4]
            ncnt = cnt + 1
            
            if ncnt > 12:
                continue
            if ne0 > target_exp[0] or ne1 > target_exp[1] or ne2 > target_exp[2] or ne3 > target_exp[3] or ne4 > target_exp[4]:
                continue
            
            new_state = (ne0, ne1, ne2, ne3, ne4, ncnt)
            if new_state not in dp[i + 1]:
                dp[i + 1][new_state] = (state, i)

    # Find a target state in dp[idx_limit]
    target_found = None
    for state in dp[idx_limit].keys():
        if state[0] == target_exp[0] and state[1] == target_exp[1] and state[2] == target_exp[2] and state[3] == target_exp[3] and state[4] == target_exp[4] and state[5] <= 12:
            target_found = state
            break
            
    if target_found is None:
        return [target]

    # Trace back
    result_indices = []
    state = target_found
    i = idx_limit
    while i > 0:
        parent = dp[i].get(state)
        if parent is None:
            i -= 1
            continue
        prev_state, cand_idx = parent
        result_indices.append(cand_idx)
        state = prev_state
        i = i - 1
        
    result_indices.reverse()
    result = [candidates[i] for i in result_indices]
    
    # Verify
    prod = 1
    for r in result:
        prod *= r
    if prod != target:
        return [target]
    if len(result) != len(set(result)):
        return [target]
    if any(r <= 12 for r in result):
        return [target]
    if len(result) > 12:
        return [target]
        
    return result
