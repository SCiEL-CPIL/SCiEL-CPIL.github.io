import time
from math import lcm

def construct():
    deadline = time.time() + 55.0
    
    nums = list(range(2, 257))
    N = len(nums)
    weights = [1.0 / x for x in nums]
    
    incompat = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            if lcm(nums[i], nums[j]) <= 256:
                incompat[i] |= (1 << j)
                incompat[j] |= (1 << i)
    
    best_sum = 0.0
    best_mask = 0
    
    def time_exceeded():
        return time.time() > deadline

    def dfs(mask, selected_mask, current_sum, sum_rem):
        nonlocal best_sum, best_mask
        
        if time_exceeded():
            return
            
        # Pruning with epsilon
        if current_sum + sum_rem < best_sum - 1e-12:
            return
            
        if mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_mask = selected_mask
            return

        # Find vertex v in mask with maximum weight
        max_w = -1
        v = -1
        m = mask
        while m:
            lb = m & (-m)
            idx = lb.bit_length() - 1
            if weights[idx] > max_w:
                max_w = weights[idx]
                v = idx
            m ^= lb
            
        if v == -1:
            return
            
        # Branch 1: Include v
        removed_mask = incompat[v] & mask
        rem_w = 0.0
        rm = removed_mask
        while rm:
            lb = rm & (-rm)
            idx = lb.bit_length() - 1
            rem_w += weights[idx]
            rm ^= lb
            
        new_sum_rem = sum_rem - weights[v] - rem_w
        new_mask = mask & ~incompat[v] & ~(1 << v)
        new_selected_mask = selected_mask | (1 << v)
        
        dfs(new_mask, new_selected_mask, current_sum + weights[v], new_sum_rem)
        
        if time_exceeded():
            return

        # Branch 2: Exclude v
        new_mask = mask & ~(1 << v)
        new_sum_rem = sum_rem - weights[v]
        
        dfs(new_mask, selected_mask, current_sum, new_sum_rem)
        
    initial_mask = (1 << N) - 1
    initial_sum_rem = sum(weights)
    
    dfs(initial_mask, 0, 0.0, initial_sum_rem)
    
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(nums[i])
            
    if not result:
        result = [256]
        
    return result
