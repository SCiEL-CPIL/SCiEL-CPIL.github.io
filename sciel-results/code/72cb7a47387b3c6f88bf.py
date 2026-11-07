import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # We search for M with many k such that k * sigma(k) = M
    # We compute sigma(k) for k up to K_max using a sieve
    
    K_max = 10**6  # Start with 1 million
    
    # Sieve for sigma
    sigma = [0] * (K_max + 1)
    for i in range(1, K_max + 1):
        for j in range(i, K_max + 1, i):
            sigma[j] += i
            
    # Map M to list of k
    M_to_ks = {}
    
    for k in range(1, K_max + 1):
        M = k * sigma[k]
        if M <= 10**18:
            if M not in M_to_ks:
                M_to_ks[M] = []
            M_to_ks[M].append(k)
            
    # Find M with most k's
    best_M = None
    best_ks = []
    max_count = 0
    
    for M, ks in M_to_ks.items():
        if len(ks) > max_count:
            max_count = len(ks)
            best_M = M
            best_ks = ks
            
    # Filter k <= sqrt(M)
    if best_M is not None:
        limit = int(math.isqrt(best_M))
        valid_ks = [k for k in best_ks if k <= limit]
    else:
        valid_ks = []
        
    # If we didn't find any, or if time allows, we could try larger K_max
    # But for now, return what we have
    # Ensure types are correct
    return {'M': best_M if best_M is not None else 1, 'k': valid_ks}
