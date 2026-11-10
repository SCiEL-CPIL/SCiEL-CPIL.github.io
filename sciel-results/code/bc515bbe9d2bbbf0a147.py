import math
from collections import defaultdict

def construct():
    # We search for M <= 10^18 with many k such that k*sigma(k) = M.
    # Strategy: Generate k from small primes, compute M = k * sigma(k), and count collisions.
    
    LIMIT_M = 10**18
    
    # Primes to include in the factorization of k
    # Using more small primes increases the chance of collisions because
    # different combinations of exponents can yield the same product M.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    
    solutions = defaultdict(list)
    best_count = 0
    best_M = 1
    best_ks = [1]

    def dfs(idx, current_k, current_M):
        nonlocal best_count, best_M, best_ks
        
        if current_M > LIMIT_M:
            return
            
        if idx == len(primes):
            # All primes considered
            if current_M > 1:
                if current_M not in solutions:
                    solutions[current_M] = []
                if current_k not in solutions[current_M]:
                    solutions[current_M].append(current_k)
                    count = len(solutions[current_M])
                    if count > best_count:
                        best_count = count
                        best_M = current_M
                        best_ks = solutions[current_M][:]
            return
        
        p = primes[idx]
        
        # Try all possible exponents a for prime p
        # T_a = p^a * sigma(p^a)
        # We need current_M * T_a <= LIMIT_M
        
        # Precompute powers and sigmas for this prime up to the limit
        # Since p is small and LIMIT_M is 10^18, a is bounded.
        # For p=2, 2^a * (2^{a+1}-1) <= 10^18 => 2^{2a+1} approx 10^18 => a approx 30.
        
        p_pow = 1
        sigma_val = 1
        for a in range(0, 60):
            if a > 0:
                p_pow *= p
                sigma_val += p_pow
                
            T = p_pow * sigma_val
            new_M = current_M * T
            if new_M > LIMIT_M:
                break
                
            new_k = current_k * p_pow
            dfs(idx + 1, new_k, new_M)

    dfs(0, 1, 1)
    
    # Ensure k_i <= sqrt(M)
    # For k > 1, sigma(k) >= k+1, so M = k*sigma(k) >= k(k+1) > k^2, so k < sqrt(M).
    # For k=1, M=1, sqrt(1)=1, k=1 is valid.
    # So the condition is always satisfied for valid M,k pairs generated this way.
    
    # Convert to list of ints
    best_ks = list(set(best_ks))
    
    return {'M': int(best_M), 'k': best_ks}
