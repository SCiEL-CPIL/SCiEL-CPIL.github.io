import math

def construct():
    # Search for M with maximum number of solutions k such that k * sigma(k) = M
    # and M <= 10^18
    # We search for k up to N
    N = 10**6  # 1,000,000
    
    # Sieve for sigma function
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
            
    # Map M to list of k's
    solutions = {}
    max_M = 10**18
    
    for k in range(1, N + 1):
        M_val = k * sigma[k]
        if M_val <= max_M:
            if M_val not in solutions:
                solutions[M_val] = []
            solutions[M_val].append(k)
            
    # Find M with most solutions
    best_M = 0
    best_k = []
    for M_val, k_list in solutions.items():
        if len(k_list) > len(best_k):
            best_k = k_list
            best_M = M_val
            
    # If we found nothing better than seed, return seed
    # Seed: M=333312, k=[336, 372, 434]
    if len(best_k) <= 3:
         return {'M': 333312, 'k': [336, 372, 434]}
    else:
         return {'M': best_M, 'k': best_k}
