"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 90s kill
    
    n = 256
    
    # Find all primes up to 256
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(n)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Precompute prime factor masks for each number 1..n
    # Mask is an integer where bit i is set if prime[i] divides the number
    factor_masks = [0] * (n + 1)
    for num in range(1, n + 1):
        mask = 0
        temp = num
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                mask |= (1 << prime_to_idx[p])
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            mask |= (1 << prime_to_idx[temp])
        factor_masks[num] = mask
    
    # Sort numbers by value descending
    # We want to maximize sum, so large numbers are prioritized
    sorted_nums = list(range(n, 0, -1))
    
    # Branch and Bound
    best_sum = 0
    best_selection = []
    
    def dfs(idx, current_sum, used_mask, current_selection):
        nonlocal best_sum, best_selection, start_time
        
        if time.time() - start_time > deadline:
            return
            
        # Pruning: if current sum + sum of all remaining possible numbers <= best_sum, prune
        # A tighter bound: sum of values of numbers from idx onwards that are coprime to used_mask
        # This is expensive to compute every time, so we use a looser bound first or precompute suffix sums
        # Let's use a simple bound: current_sum + sum(sorted_nums[idx:])
        # If this is <= best_sum, we can prune.
        # To make it tighter, we can precompute suffix sums of sorted_nums.
        if current_sum + suffix_sums[idx] <= best_sum:
            return
            
        if idx == len(sorted_nums):
            if current_sum > best_sum:
                best_sum = current_sum
                best_selection = list(current_selection)
            return
        
        num = sorted_nums[idx]
        
        # Option 1: Include num
        if (used_mask & factor_masks[num]) == 0:
            current_selection.append(num)
            dfs(idx + 1, current_sum + num, used_mask | factor_masks[num], current_selection)
            current_selection.pop()
            
        # Option 2: Exclude num
        dfs(idx + 1, current_sum, used_mask, current_selection)
        
    # Precompute suffix sums for bounding
    suffix_sums = [0] * (len(sorted_nums) + 1)
    for i in range(len(sorted_nums) - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + sorted_nums[i]
        
    dfs(0, 0, 0, [])
    
    return best_selection

# EVOLVE-BLOCK-END
