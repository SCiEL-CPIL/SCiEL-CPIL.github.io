"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin before 120s limit

    # Primes to use for generating smooth numbers
    # Using primes up to 47
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n_primes = len(primes)
    
    max_k = 10**9
    max_m = 10**18
    
    # Dictionary to store lists of k for each M
    m_to_ks = defaultdict(list)
    
    # Precompute prime powers and sigma contributions for each prime
    # For prime p, we store: p^0, p^1, ..., p^e and corresponding sigma contributions
    # sigma(p^0) = 1, sigma(p^1) = 1+p, sigma(p^2) = 1+p+p^2, ...
    prime_powers = []  # prime_powers[i] = list of (p^e, sigma(p^e))
    for p in primes:
        powers = []
        p_pow = 1
        sigma_p = 1
        for e in range(30):
            powers.append((p_pow, sigma_p))
            p_pow *= p
            if p_pow > max_k:
                break
            sigma_p += p_pow
        prime_powers.append(powers)
    
    # Use iterative approach with explicit stack to avoid recursion depth issues
    # State: (idx, k, sigma_k)
    # We'll use a stack-based DFS
    
    def time_check():
        return time.time() > deadline
    
    stack = [(0, 1, 1)]
    count = 0
    
    while stack:
        if time_check():
            break
        
        idx, k, sigma_k = stack.pop()
        count += 1
        
        if idx == n_primes:
            # We have considered all primes, k is a valid smooth number
            M = k * sigma_k
            if M <= max_m and k <= max_k:
                m_to_ks[M].append(k)
            continue
        
        p = primes[idx]
        # Iterate over possible exponents for prime p
        powers = prime_powers[idx]
        
        for p_pow, sigma_p in powers:
            new_k = k * p_pow
            if new_k > max_k:
                break
            
            new_sigma = sigma_k * sigma_p
            
            # Prune if M exceeds limit
            if new_k * new_sigma > max_m:
                break
            
            stack.append((idx + 1, new_k, new_sigma))
    
    # Find the best M
    best_m = 0
    best_ks = []
    for M, ks in m_to_ks.items():
        if len(ks) > len(best_ks):
            best_m = M
            best_ks = ks
            
    if not best_ks:
        # Fallback to known solution
        return {'M': 333312, 'k': [336, 372, 434]}
        
    # Ensure types are Python int
    best_ks = [int(k) for k in best_ks]
    
    return {'M': int(best_m), 'k': best_ks}
# EVOLVE-BLOCK-END
