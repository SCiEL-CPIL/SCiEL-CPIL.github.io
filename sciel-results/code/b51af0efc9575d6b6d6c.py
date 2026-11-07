import time
import math

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Sieve for sigma(k)
    N = 1_000_000
    sigma = [1] * (N + 1)
    
    for i in range(2, N + 1):
        if sigma[i] == 1:  # i is prime
            for j in range(i, N + 1, i):
                # Calculate the contribution of prime i to sigma(j)
                # Let j = i^a * m where m is not divisible by i
                # The factor for i is 1 + i + ... + i^a
                temp = j
                power = 1
                sum_power = 1
                while temp % i == 0:
                    temp //= i
                    power *= i
                    sum_power += power
                sigma[j] *= sum_power
                
    # Group k by M = k * sigma(k)
    # We only care about M <= 10^18
    groups = {}
    for k in range(1, N + 1):
        M = k * sigma[k]
        if M <= 10**18:
            if M not in groups:
                groups[M] = []
            groups[M].append(k)
            
    # Find the M with the most k's
    best_M = 1
    best_k = []
    for M, k_list in groups.items():
        if len(k_list) > len(best_k):
            best_M = M
            best_k = k_list
            
    # Sort the k's as required (distinct, list)
    best_k.sort()
    
    return {'M': int(best_M), 'k': [int(k) for k in best_k]}
