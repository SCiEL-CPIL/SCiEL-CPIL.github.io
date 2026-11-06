import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # Helper functions for verification
    
    def _subset_mean_exists(selected, x):
        n = len(selected)
        if n < 2:
            return False
        
        max_sum = sum(selected)
        
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for elem in selected:
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k + 1] |= (dp[k] << elem)
        
        for k in range(2, n + 1):
            target = k * x
            if target > max_sum:
                break
            if (dp[k] >> target) & 1:
                return True
        
        return False

    def _subset_sum_with_target_exists(others, a, x):
        n = len(others)
        if n == 0:
            return False
        
        max_sum = sum(others)
        
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for elem in others:
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k + 1] |= (dp[k] << elem)
        
        for k in range(1, n + 1):
            target = (k + 1) * a - x
            if target < 0:
                continue
            if target > max_sum:
                continue
            if (dp[k] >> target) & 1:
                return True
        
        return False

    def _is_valid_addition(selected, x):
        n = len(selected)
        
        if n >= 2:
            if _subset_mean_exists(selected, x):
                return False
        
        for a in selected:
            others = [y for y in selected if y != a]
            if not others:
                continue
            if _subset_sum_with_target_exists(others, a, x):
                return False
        
        return True

    # Step 1: Generate base-3 0/1 set
    def generate_base3_set(limit):
        s = set()
        powers = []
        p = 1
        while p <= limit:
            powers.append(p)
            p *= 3
        
        n = len(powers)
        for mask in range(1 << n):
            val = 0
            for i in range(n):
                if mask & (1 << i):
                    val += powers[i]
            if val <= limit:
                s.add(val)
        return list(s)
    
    base_set = generate_base3_set(512)
    
    # Step 2: Clean the base set
    current_set = list(base_set)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(current_set)):
            x = current_set[i]
            others = [y for j, y in enumerate(current_set) if j != i]
            if _subset_mean_exists(others, x):
                current_set.pop(i)
                changed = True
                break
    
    best_set = current_set
    
    # Step 3: If size >= 48, return 48
    if len(best_set) >= 48:
        return best_set[:48]
    
    # Step 4: Try to add more elements greedily
    candidates = [x for x in range(1, 513) if x not in set(best_set)]
    random.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline:
            break
        if len(best_set) >= 48:
            break
        
        if _is_valid_addition(best_set, x):
            best_set.append(x)
    
    return best_set

