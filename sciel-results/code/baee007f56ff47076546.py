# EVOLVE-BLOCK-START
import sys
from math import gcd

def construct():
    N = 256
    
    # Generate primes up to N
    sieve = [True] * (N + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(N**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, N + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, N + 1) if sieve[i]]
    
    # Map prime to index for bitmask
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Precompute prime factor masks for each number
    factor_masks = [0] * (N + 1)
    for i in range(2, N + 1):
        mask = 0
        x = i
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                mask |= (1 << prime_to_idx[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            mask |= (1 << prime_to_idx[x])
        factor_masks[i] = mask
    
    # 1 has no prime factors
    factor_masks[1] = 0
    
    # Sort numbers in descending order to find high-value solutions early
    nums = list(range(1, N + 1))
    nums.sort(reverse=True)
    
    # Precompute suffix sums for pruning
    suffix_sum = [0] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + nums[i]
        
    # Greedy initialization for a strong lower bound
    greedy_set = []
    greedy_mask = 0
    for x in nums:
        mask = factor_masks[x]
        if (greedy_mask & mask) == 0:
            greedy_set.append(x)
            greedy_mask |= mask
            
    best_sum = sum(greedy_set)
    best_set = greedy_set[:]
    
    # Increase recursion limit just in case
    sys.setrecursionlimit(10000)
    
    def search(idx, current_sum, used_mask, current_set):
        nonlocal best_sum, best_set
        
        # Pruning
        if current_sum + suffix_sum[idx] <= best_sum:
            return
            
        if idx == N:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = current_set[:]
            return
            
        x = nums[idx]
        mask = factor_masks[x]
        
        # Option 1: Skip x
        search(idx + 1, current_sum, used_mask, current_set)
        
        # Option 2: Take x if compatible
        if (used_mask & mask) == 0:
            current_set.append(x)
            search(idx + 1, current_sum + x, used_mask | mask, current_set)
            current_set.pop()
            
    search(0, 0, 0, [])
    
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
