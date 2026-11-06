import numpy as np
import random
import time

def construct():
    start_time = time.time()
    max_time = 85.0
    n = 512
    max_size = 48
    max_sum = max_size * n + 100

    best_set = []
    best_size = 0

    def can_add(x, S, dp, current_max_k):
        for k in range(2, current_max_k + 1):
            target = k * x
            if target > max_sum:
                break
            if dp[k][target]:
                return False
        for y in S:
            for k in range(1, current_max_k + 1):
                target = (k + 1) * y - x
                if target < 0 or target > max_sum:
                    continue
                if dp[k][target]:
                    return False
        return True

    def update_dp(dp, x, current_max_k):
        for k in range(current_max_k, -1, -1):
            if x <= max_sum:
                dp[k+1][x:] |= dp[k][:-x]
        return current_max_k + 1

    def greedy_construct(S, dp, current_max_k, candidates, time_limit):
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > time_limit:
                break
            if len(S) >= max_size:
                break
            if can_add(x, S, dp, current_max_k):
                current_max_k = update_dp(dp, x, current_max_k)
                S.append(x)
        return S, dp, current_max_k

    for restart in range(2000):
        if time.time() - start_time > max_time:
            break

        S = []
        dp = [np.zeros(max_sum + 1, dtype=bool) for _ in range(max_size + 1)]
        dp[0][0] = True
        current_max_k = 0

        candidates = list(range(1, n + 1))
        S, dp, current_max_k = greedy_construct(S, dp, current_max_k, candidates, max_time - 5.0)

        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]

        if len(S) == max_size:
            return best_set

        # Local Search
        # Try to improve S by swapping elements
        local_time_limit = min(start_time + max_time - 2.0, time.time() + 2.0)
        
        improved = False
        while time.time() < local_time_limit and not improved:
            if len(S) >= max_size:
                break
                
            # Pick a random element to remove
            i = random.randint(0, len(S) - 1)
            x_remove = S[i]
            
            S_temp = S[:i] + S[i+1:]
            
            # Build DP for S_temp
            dp_temp = [np.zeros(max_sum + 1, dtype=bool) for _ in range(max_size + 1)]
            dp_temp[0][0] = True
            k_temp = 0
            for y in S_temp:
                k_temp = update_dp(dp_temp, y, k_temp)
                
            # Try to add a new element
            # We try a few random candidates
            found_swap = False
            for _ in range(50):
                if time.time() > local_time_limit:
                    break
                    
                y_add = random.randint(1, n)
                if y_add in S_temp:
                    continue
                
                if can_add(y_add, S_temp, dp_temp, k_temp):
                    S_new = S_temp + [y_add]
                    k_temp = update_dp(dp_temp, y_add, k_temp)
                    
                    # Try to add more
                    S_set = set(S_new)
                    cand_list = [z for z in range(1, n + 1) if z not in S_set]
                    
                    S_new, dp_temp, k_temp = greedy_construct(S_new, dp_temp, k_temp, cand_list, local_time_limit)
                    
                    if len(S_new) > len(S):
                        S = S_new
                        best_size = len(S)
                        best_set = S[:]
                        improved = True
                        break
                    else:
                        # If we didn't improve, we might still want to accept the swap 
                        # to explore the neighborhood? 
                        # For now, we only accept improvements to keep it simple.
                        pass

    return best_set
