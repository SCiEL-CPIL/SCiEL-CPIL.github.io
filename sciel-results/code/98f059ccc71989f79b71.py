import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # We want to find M with many solutions k.
    # M = k * sigma(k)
    # We search for k up to K_max.
    # Larger K_max increases the chance of finding M with more solutions.
    
    # Estimate K_max based on time budget
    # Sieve O(N log N) + Loop O(N)
    # Let's try N = 10^7. 
    # If time allows, we could go higher, but 10^7 is a solid baseline.
    K_max = 10_000_000
    
    # 1. Compute sigma(k) for k in [1, K_max]
    sigma = [0] * (K_max + 1)
    for i in range(1, K_max + 1):
        for j in range(i, K_max + 1, i):
            sigma[j] += i
            
    # 2. Map M to list of k
    # M = k * sigma(k)
    # We store counts and the list of k's
    # To save memory, we might just store counts first, then re-compute?
    # No, re-computing is O(N) again, which is fine.
    # But storing lists is better if we don't want to re-scan.
    # Number of distinct M is roughly K_max / log(K_max) or similar?
    # Actually, M values are sparse.
    # Let's just store the list.
    
    solutions = {}
    
    for k in range(1, K_max + 1):
        if time.time() > deadline:
            break
        M = k * sigma[k]
        if M > 10**18:
            continue
        if M not in solutions:
            solutions[M] = []
        solutions[M].append(k)
        
    # 3. Find M with max number of solutions
    best_M = 0
    best_k_list = []
    
    for M, k_list in solutions.items():
        if len(k_list) > len(best_k_list):
            best_M = M
            best_k_list = k_list
            
    # If we didn't find anything (unlikely), fallback to hardcoded
    if not best_k_list:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    # Ensure types are correct
    # M is int, k_list is list of ints
    return {'M': int(best_M), 'k': list(best_k_list)}
