import math
import time
from itertools import product

def construct():
    start_time = time.time()
    N = 479001600  # 12!
    
    # Prime factorization of 12!: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # We need to partition these into k groups (factors), each > 12, distinct, minimizing max
    
    best_factors = [N]
    best_max_val = N
    
    def deadline_exceeded():
        return time.time() - start_time > 80
    
    # Generate all divisors of N that are > 12
    # But this is too many. Instead, work with prime exponent vectors.
    
    # Strategy: For each k from 7 down to 2, try to find k distinct divisors > 12 
    # whose product is N, minimizing the max.
    # 
    # Better: enumerate all valid "factor multisets" by assigning prime powers to slots.
    # Since we have primes [2,3,5,7,11] with exponents [10,5,2,1,1],
    # each factor is determined by how much of each prime it gets.
    #
    # For a given k, we need to distribute the exponents among k slots such that:
    # - Each slot gets a non-negative integer exponent for each prime
    # - Sum of exponents for prime p across all slots = exponent of p in N
    # - Product of exponents assigned to each slot > 12
    # - All resulting products are distinct
    # - Minimize the maximum product
    #
    # This is still complex. Let me try a direct DFS on divisors but with better pruning.
    
    # Precompute all divisors of N
    divisors = []
    for i in range(1, int(N**0.5) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    divisors = [d for d in divisors if d > 12]
    divisors.sort()
    
    # For each k, do a DFS picking k distinct divisors from the list
    # such that their product is N, minimizing the max.
    # 
    # Key optimization: sort divisors, use the fact that if we pick factors in increasing order,
    # the last factor is the max. We want to minimize this.
    # 
    # Use a recursive search with strong pruning:
    # - At depth d, we've picked d factors (in increasing order), product so far = prod
    # - Remaining product needed = N // prod
    # - We need (k-d) more factors, each > last_picked, distinct, product = N//prod
    # - The smallest possible product for (k-d) factors > last_picked is (last+1)(last+2)...(last+k-d-1)
    # - If N//prod < that, prune
    # - Also, the largest factor we'll pick is at most N//prod (if it's the last one)
    #   or smaller if we need multiple more factors
    
    # Let's try k from 7 down to 2, and for each k, do the DFS.
    # Stop early if we find a solution with max < current best.
    
    for k in range(7, 1, -1):
        if deadline_exceeded():
            break
        
        # DFS: pick k factors in increasing order from divisors
        def dfs(d, last, prod, path):
            nonlocal best_max_val, best_factors
            
            if deadline_exceeded():
                return
            
            if d == k:
                if prod == N:
                    max_f = path[-1]  # since sorted, last is max
                    if max_f < best_max_val:
                        best_max_val = max_f
                        best_factors = list(path)
                return
            
            remaining = k - d
            target_rem = N // prod if N % prod == 0 else None
            if target_rem is None:
                return
            
            # Minimum product for remaining factors (each > last, distinct)
            min_prod = 1
            for i in range(1, remaining + 1):
                min_prod *= (last + i)
            if target_rem < min_prod:
                return
            
            # If remaining == 1, the next factor must be target_rem
            if remaining == 1:
                x = target_rem
                if x > last and x > 12 and x in divisors:
                    # Check if x is actually a divisor (it should be by construction)
                    path.append(x)
                    dfs(d + 1, x, prod * x, path)
                    path.pop()
                return
            
            # For remaining > 1, iterate through possible next factors
            # x must be > last, x divides target_rem, and x is in our divisor list
            # Upper bound: x * (x+1) * ... * (x+remaining-1) <= target_rem
            # So x <= target_rem^(1/remaining) roughly
            x_upper = int(target_rem ** (1.0 / remaining)) + 20
            
            # Find divisors in (last, x_upper]
            # Use binary search or just iterate
            for x in divisors:
                if x <= last:
                    continue
                if x > x_upper:
                    break
                if target_rem % x != 0:
                    continue
                
                path.append(x)
                dfs(d + 1, x, prod * x, path)
                path.pop()
        
        dfs(0, 12, 1, [])
        
        # If we found a good solution, we can stop searching smaller k
        # since more factors tend to give smaller max
        if best_max_val < N:
            # We found something better than the seed
            # Check if we should continue or stop
            # For now, continue to see if we can do better with smaller k
            pass
    
    return best_factors
