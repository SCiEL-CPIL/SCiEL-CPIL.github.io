import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    target = Fraction(5, 31)
    
    # Fallback solution
    best_solution = [8, 31, 248]
    best_max = 248

    # Structural Change: Exhaustive search for small M
    # We iterate M from 1 upwards.
    # For each M, we check if there exists a subset of {1..M} 
    # whose reciprocals sum to 5/31.
    # We stop at the first M that works. This guarantees minimal M.
    
    # To avoid combinatorial explosion, we limit the number of terms k.
    # Min terms is 3.
    # Max terms? If M is small, k can be larger.
    # If M=10, max sum is 1+1/2+...+1/10 ~ 2.9.
    # We need sum = 0.16.
    # So k won't be huge.
    
    # We will iterate M from 1 to 248.
    # If M is small, combinations are cheap.
    # If M gets large, we might time out.
    # We check time in the inner loop.
    
    for M in range(1, 249):
        if time.time() > deadline:
            break
            
        # For this M, try all k from 3 to M
        # Optimization: 
        # If sum(1/1 + ... + 1/M) < 5/31, continue to next M.
        # Actually, we are increasing M, so the max possible sum increases.
        # Once max_sum >= 5/31, we can start checking.
        
        max_possible_sum = sum(Fraction(1, i) for i in range(1, M+1))
        if max_possible_sum < target:
            continue
            
        # Try different k
        # k must be at least 3.
        # k cannot be larger than M.
        # Also, if k is too large, the sum of the smallest k reciprocals might exceed target?
        # No, we are picking ANY k distinct from 1..M.
        # The smallest sum for k terms is 1/M + 1/(M-1) + ...
        # The largest sum for k terms is 1/1 + 1/2 + ...
        
        # We iterate k from 3 upwards.
        for k in range(3, M + 1):
            if time.time() > deadline:
                break
                
            # Pruning: 
            # Min sum for k terms from 1..M is sum(1/M, 1/(M-1), ..., 1/(M-k+1))
            # Max sum for k terms from 1..M is sum(1/1, 1/2, ..., 1/k)
            
            min_sum = sum(Fraction(1, M - i) for i in range(k))
            max_sum_k = sum(Fraction(1, i) for i in range(1, k+1))
            
            if min_sum > target or max_sum_k < target:
                continue
                
            # Enumerate combinations
            # combinations(range(1, M+1), k)
            for comb in combinations(range(1, M+1), k):
                if time.time() > deadline:
                    break
                    
                s = sum(Fraction(1, d) for d in comb)
                if s == target:
                    return list(comb)
                    
    return best_solution
