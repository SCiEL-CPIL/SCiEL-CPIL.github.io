import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    max_m = 10**18
    
    # Structural change: Use a sieve to compute k*sigma(k) for a range of k.
    # This is a different search method than recursive smooth number generation.
    # Knowledge suggests L=10^6 yields score ~1.667.
    L = 10**6
    
    # Sieve for sigma(k)
    sigma = [0] * (L + 1)
    
    # Compute sum of divisors for all numbers up to L
    # O(L log L)
    for d in range(1, L + 1):
        for m in range(d, L + 1, d):
            sigma[m] += d
            
    # Map M = k * sigma(k) to list of k
    solutions = {}
    
    for k in range(1, L + 1):
        m_val = k * sigma[k]
        if m_val <= max_m:
            if m_val not in solutions:
                solutions[m_val] = []
            solutions[m_val].append(k)
            
    best_m = 0
    best_k_list = []
    
    # Find the M with the most solutions
    for m, k_list in solutions.items():
        if len(k_list) > len(best_k_list):
            best_m = m
            best_k_list = k_list
            
    # Fallback if no solutions found
    if not best_k_list:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    # Ensure k_i <= sqrt(M)
    # k <= sqrt(k*sigma(k)) is equivalent to k <= sigma(k), which is true for all k >= 1
    # But we verify just in case
    sqrt_m = math.isqrt(best_m)
    valid_k = [k for k in best_k_list if k <= sqrt_m]
    valid_k.sort()
    
    M_out = int(best_m)
    k_out = [int(k) for k in valid_k]
    
    return {'M': M_out, 'k': k_out}

if __name__ == "__main__":
    result = construct()
    print(result)
