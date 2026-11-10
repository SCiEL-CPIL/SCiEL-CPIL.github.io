import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    solutions = defaultdict(list)
    node_count = 0
    max_nodes = 5_000_000 # Safety limit
    
    def dfs(idx, k, sigma):
        nonlocal node_count
        if time.time() > deadline or node_count > max_nodes:
            return
        
        node_count += 1
        
        if idx == len(primes):
            if k > 1:
                M = k * sigma
                if M <= 10**18:
                    solutions[M].append(k)
            return
        
        p = primes[idx]
        
        # Option 1: Exclude p
        dfs(idx + 1, k, sigma)
        
        # Option 2: Include p^a
        pk = 1
        sigma_factor = 1
        for a in range(1, 100):
            pk *= p
            # Check overflow for M
            # M = k * pk * sigma * sigma_factor
            # We need M <= 10^18
            
            # Update sigma factor
            if a == 1:
                sigma_factor = 1 + p
            else:
                sigma_factor = sigma_factor * p + 1
                
            # Early break if k * pk is too big
            if k > 10**18 // pk:
                break
            
            new_k = k * pk
            # Check M limit
            # M = new_k * sigma * sigma_factor
            # To avoid overflow in intermediate steps, check bounds
            if sigma > 10**18 // (new_k * sigma_factor):
                break
                
            new_sigma = sigma * sigma_factor
            M = new_k * new_sigma
            
            if M > 10**18:
                break
                
            solutions[M].append(new_k)
            
            # Recurse
            dfs(idx + 1, new_k, new_sigma)
            
            if time.time() > deadline or node_count > max_nodes:
                return

    dfs(0, 1, 1)
    
    # Find best
    best_M = -1
    best_ks = []
    
    for M, ks in solutions.items():
        if len(ks) > len(best_ks):
            best_M = M
            best_ks = ks
            
    if best_M == -1:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    # Verify
    def sigma(n):
        res = 1
        d = 2
        while d * d <= n:
            if n % d == 0:
                term = 1
                while n % d == 0:
                    n //= d
                    term *= d
                res *= (term * d - 1) // (d - 1)
            d += 1
        if n > 1:
            res *= (n + 1)
        return res

    valid_ks = []
    for k in best_ks:
        if k * k <= best_M and k * sigma(k) == best_M:
            valid_ks.append(k)
            
    if not valid_ks:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': best_M, 'k': valid_ks}
