import time
from math import factorial
from functools import lru_cache

TARGET = factorial(12)
PRIMES = [2, 3, 5, 7, 11]
EXPS = [10, 5, 2, 1, 1]

def _get_divisors():
    divs = []
    def gen(idx, cur):
        if idx == len(PRIMES):
            if cur > 12:
                divs.append(cur)
            return
        p = PRIMES[idx]
        for e in range(EXPS[idx] + 1):
            gen(idx + 1, cur * (p ** e))
    gen(0, 1)
    divs.sort()
    return divs

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    all_divs = _get_divisors()
    n_divs = len(all_divs)
    
    # We iterate k from 0 to n_divs-1
    # For each k, we check if a subset of all_divs[0:k+1] multiplies to TARGET
    
    # Optimization: If the product of the largest 12 divisors in [0:k] is < TARGET, skip.
    # Actually, we can just let DFS fail.
    
    for k in range(n_divs):
        if time.time() > deadline - 5.0:
            break
            
        # The candidate set is all_divs[0:k+1]
        # We need to find a subset of this list with product TARGET.
        # Since we are increasing k, the first time we find one, the max factor is all_divs[k].
        
        # Define DFS
        # Memo key: (start_idx, target)
        # start_idx is index in all_divs
        # We only consider indices <= k
        
        memo = {}
        
        def dfs(idx, target):
            if time.time() > deadline - 1.0:
                return None
            
            if target == 1:
                return []
            
            if idx > k:
                return None
                
            if (idx, target) in memo:
                return memo[(idx, target)]
            
            # Pruning: If target is not divisible by any possible next factor, we can't proceed?
            # We just iterate.
            
            for i in range(idx, k + 1):
                d = all_divs[i]
                if d > target:
                    break # Since sorted ascending, if d > target, larger d's also > target
                    # Wait, if d > target, we can't use d to divide target (unless target is 1, handled above).
                    # Actually, if d > target, we can't use d.
                    # But we might use smaller ones? No, we are iterating i from idx upwards.
                    # If d > target, then for all j > i, all_divs[j] >= d > target.
                    # So we can break.
                    
                if target % d != 0:
                    continue
                    
                sub = dfs(i + 1, target // d)
                if sub is not None:
                    res = [d] + sub
                    memo[(idx, target)] = res
                    return res
            
            memo[(idx, target)] = None
            return None

        result = dfs(0, TARGET)
        if result is not None:
            return result
            
    # Fallback
    return [TARGET]
