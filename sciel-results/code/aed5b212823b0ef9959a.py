import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Attempt to find minimal k via DFS
    for k in range(9, 15):
        if time.time() > deadline:
            break
        sol = find_min_solution(k)
        if sol:
            return sol
            
    # Fallback: Greedy algorithm
    return greedy_solution()

def find_min_solution(k):
    target = Fraction(1, 1)
    
    def dfs(start_denom, remaining_sum, terms_left, current_list):
        if time.time() > deadline:
            return None
        if terms_left == 0:
            return current_list if remaining_sum == 0 else None
        
        # Pruning 1: Max possible sum
        max_possible = sum(Fraction(1, i) for i in range(start_denom, start_denom + terms_left))
        if remaining_sum > max_possible:
            return None
            
        # Determine min d
        # 1/d <= remaining_sum => d >= 1/remaining_sum
        min_d = max(start_denom, int(1 / remaining_sum) + 1)
        
        # Iterate d
        # We can limit the number of iterations if it gets too deep, but k is small
        for d in range(min_d, min_d + 5000):
            if time.time() > deadline:
                return None
            term = Fraction(1, d)
            if term > remaining_sum:
                break # d is too small, but we started at min_d, so this shouldn't happen unless int truncation was bad
            new_remaining = remaining_sum - term
            
            if terms_left == 1:
                if new_remaining == 0:
                    return current_list + [d]
                else:
                    continue # d needs to be larger? No, if new_rem != 0, this d is wrong.
                    # Actually if terms_left==1, we MUST have new_remaining == 0.
                    # So if it's not 0, we just move to next d?
                    # But 1/d is decreasing. If new_remaining > 0, we need larger 1/d -> smaller d.
                    # But we are increasing d. So if new_remaining > 0, it will stay > 0?
                    # No. new_remaining = R - 1/d. As d increases, 1/d decreases, new_remaining increases.
                    # So if new_remaining > 0, it will get bigger. We want 0.
                    # So if at min_d, new_remaining > 0, then for all larger d, new_remaining > 0.
                    # So we can break?
                    # Wait. If new_remaining < 0, we broke earlier.
                    # So if new_remaining != 0, it must be > 0.
                    # So we can break the loop.
                    break

            # Check if new_remaining is reachable
            max_next = sum(Fraction(1, i) for i in range(d + 1, d + terms_left))
            if new_remaining > max_next:
                # As d increases, new_remaining increases and max_next decreases.
                # So the condition will hold for all larger d.
                break
                
            res = dfs(d + 1, new_remaining, terms_left - 1, current_list + [d])
            if res:
                return res
        return None

    return dfs(5, target, k, [])

def greedy_solution():
    res = []
    rem = Fraction(1, 1)
    last = 4
    while rem > 0:
        # Find smallest d > last such that 1/d <= rem
        # d >= 1/rem
        d = max(last + 1, int(1 / rem) + 1)
        # Check if 1/d <= rem
        while Fraction(1, d) > rem:
            d += 1
        res.append(d)
        rem -= Fraction(1, d)
        last = d
        if len(res) > 100: # Safety break
            break
    return res
