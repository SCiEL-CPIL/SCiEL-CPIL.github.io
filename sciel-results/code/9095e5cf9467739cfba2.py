import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    N = 512
    
    # Sieve for primes up to 511
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return is_prime
    
    is_prime = sieve(511)
    primes = [p for p in range(2, 512) if is_prime[p]]
    
    # Precompute unique prime factors for 1..511
    factors = [()] * 512
    factors[1] = ()
    for i in range(2, 512):
        temp = i
        facs = []
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                facs.append(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            facs.append(temp)
        factors[i] = tuple(facs)

    # Items: 1..511
    items = list(range(1, 512))
    weights = [1.0 / (512 - a) for a in items]
    
    # Sort by weight descending
    sorted_indices = sorted(range(len(items)), key=lambda i: weights[i], reverse=True)
    
    sorted_items = [items[i] for i in sorted_indices]
    sorted_weights = [weights[i] for i in sorted_indices]
    sorted_factors = [factors[i] for i in sorted_indices]
    
    n_items = len(sorted_items)
    
    # Suffix sums for upper bound
    suffix_sum = [0.0] * (n_items + 1)
    for i in range(n_items - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i+1] + sorted_weights[i]
        
    best_score = 0.0
    best_selection = [False] * n_items
    
    def dfs(idx, current_score, used_primes, selection):
        nonlocal best_score, best_selection
        
        if time.time() > deadline:
            return
            
        # Prune
        if current_score + suffix_sum[idx] <= best_score:
            return
            
        if idx == n_items:
            if current_score > best_score:
                best_score = current_score
                best_selection = selection[:]
            return
            
        # Try to include item idx
        facs = sorted_factors[idx]
        # Check conflict
        can_include = True
        for p in facs:
            if p in used_primes:
                can_include = False
                break
                
        if can_include:
            selection[idx] = True
            dfs(idx + 1, current_score + sorted_weights[idx], used_primes | set(facs), selection)
            selection[idx] = False
            
        # Try to exclude item idx
        dfs(idx + 1, current_score, used_primes, selection)
        
    dfs(0, 0.0, set(), [False] * n_items)
    
    result = []
    for i in range(n_items):
        if best_selection[i]:
            result.append(sorted_items[i])
            
    return result
