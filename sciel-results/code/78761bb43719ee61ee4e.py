import time
import math
import sys

sys.setrecursionlimit(10000)

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors of 5040
    divisors = [i for i in range(1, n + 1) if n % i == 0]
    divisors.sort(reverse=True)
    m = len(divisors)
    
    # Precompute suffix sums for max sum pruning
    # suffix_sum[i] = sum(divisors[i:])
    suffix_sum = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors[i]
        
    # Precompute prefix sums for min sum pruning?
    # Actually, for min sum from idx with count items:
    # We want the smallest 'count' items in divisors[idx:].
    # Since divisors is sorted descending, the smallest items are at the end.
    # The smallest 'count' items in divisors[idx:] are divisors[m-count : m]
    # provided m-count >= idx.
    # Sum = suffix_sum[m-count] - suffix_sum[m] = suffix_sum[m-count]
    # (since suffix_sum[m] = 0)
    
    def min_sum_from(idx, count):
        if m - count < idx:
            return float('inf')
        return suffix_sum[m - count]
    
    start_time = time.time()
    deadline = start_time + 80.0
    
    # We want to minimize the number of divisors k.
    # Iterate k from 1 upwards.
    
    # Upper bound for k:
    # The sum of the k smallest divisors must be <= target.
    # The smallest divisors are 1, 2, ...
    # Sum of first k integers is k(k+1)/2.
    # k(k+1)/2 <= 4979 => k^2 ~ 10000 => k ~ 100.
    # So k is likely small.
    
    # Lower bound for k:
    # Sum of top k divisors must be >= target.
    
    for k in range(1, m + 1):
        # Prune k if impossible
        # Max sum with k items: sum of largest k
        max_possible = suffix_sum[0] - suffix_sum[k]
        if max_possible < target:
            continue
            
        # Min sum with k items: sum of smallest k
        min_possible = min_sum_from(0, k)
        if min_possible > target:
            # If the sum of the smallest k items is already > target,
            # then any set of k items will have sum > target.
            # Since min sum increases with k, we can break.
            break
            
        # DFS to find a subset of size k summing to target
        path = []
        found = False
        
        def dfs(idx, remaining, count):
            nonlocal found
            if found:
                return
            if time.time() > deadline:
                return
                
            if count == 0:
                if remaining == 0:
                    found = True
                return
            if idx >= m:
                return
                
            # Pruning: Max sum
            # Max sum we can get from divisors[idx:] with 'count' items
            # is the sum of the first 'count' items in that range:
            # divisors[idx], ..., divisors[idx+count-1]
            if idx + count <= m:
                max_sum_here = suffix_sum[idx] - suffix_sum[idx + count]
                if remaining > max_sum_here:
                    return
            else:
                # Not enough items left to pick 'count'
                return

            # Pruning: Min sum
            # Min sum we can get from divisors[idx:] with 'count' items
            min_sum_here = min_sum_from(idx, count)
            if remaining < min_sum_here:
                return
            
            # Option 1: Pick divisors[idx]
            if divisors[idx] <= remaining:
                path.append(divisors[idx])
                dfs(idx + 1, remaining - divisors[idx], count - 1)
                path.pop()
                
            if found:
                return
                
            # Option 2: Skip divisors[idx]
            # Ensure we have enough items left to pick 'count'
            if m - (idx + 1) >= count:
                dfs(idx + 1, remaining, count)
                
        dfs(0, target, k)
        
        if found:
            if sum(path) == target:
                return path

    # Fallback: Greedy (should not be reached if a solution exists)
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    return chosen
