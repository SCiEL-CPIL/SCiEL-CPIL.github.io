# EVOLVE-BLOCK-START
import time
from math import isqrt

def _precompute():
    n = 256
    pf = [[] for _ in range(n + 1)]
    for i in range(2, n + 1):
        if not pf[i]:
            for j in range(i, n + 1, i):
                pf[j].append(i)
    return pf

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 256
    pf = _precompute()
    
    nums = list(range(n, 1, -1))
    num_pf = {m: pf[m] for m in nums}
    
    all_primes = set()
    for m in nums:
        for p in num_pf[m]:
            all_primes.add(p)
    prime_list = sorted(all_primes)
    prime_to_idx = {p: i for i, p in enumerate(prime_list)}
    num_prime_mask = {}
    for m in nums:
        mask = 0
        for p in num_pf[m]:
            mask |= (1 << prime_to_idx[p])
        num_prime_mask[m] = mask
    
    suffix_sum = [0] * (len(nums) + 1)
    for i in range(len(nums) - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + nums[i]
    
    # Greedy initial solution
    used_mask = 0
    greedy_sum = 0
    greedy_set = []
    for m in nums:
        mask = num_prime_mask[m]
        if mask & used_mask == 0:
            used_mask |= mask
            greedy_sum += m
            greedy_set.append(m)
    
    best_sum = greedy_sum
    best_set = set(greedy_set)
    
    import sys
    sys.setrecursionlimit(10000)
    
    def dfs(idx, used_mask, current_sum, current_set):
        nonlocal best_sum, best_set, deadline
        
        if time.time() > deadline:
            return
        
        if idx == len(nums):
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(current_set)
            return
        
        if current_sum + suffix_sum[idx] <= best_sum:
            return
        
        m = nums[idx]
        mask = num_prime_mask[m]
        
        # Try include first (since we sorted descending, larger values first)
        if mask & used_mask == 0:
            dfs(idx + 1, used_mask | mask, current_sum + m, current_set + [m])
            if time.time() > deadline:
                return
        
        # Try exclude
        dfs(idx + 1, used_mask, current_sum, current_set)
    
    dfs(0, 0, 0, [])
    
    result = [1] + sorted(best_set)
    return result
# EVOLVE-BLOCK-END
