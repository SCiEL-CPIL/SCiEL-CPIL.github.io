import time
from math import gcd

def construct():
    start_time = time.time()
    # Use a safe deadline, e.g., 120s out of 150s
    deadline = start_time + 120.0
    
    # 1. Generate primes < 512
    primes = []
    for i in range(2, 512):
        is_prime = True
        for p in primes:
            if p * p > i:
                break
            if i % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
            
    # 2. Initialize groups
    # Each group is a list of primes
    groups = [[p] for p in primes]
    
    # Helper to get product
    def prod(lst):
        r = 1
        for x in lst:
            r *= x
        return r
    
    # Helper to get score
    def calc_score(groups):
        s = 0.0
        for g in groups:
            p = prod(g)
            if p < 512:
                s += 1.0 / (512.0 - p)
        return s
    
    # 3. Greedy Merging Loop
    while time.time() < deadline:
        best_gain = 0.0
        best_i, best_j = -1, -1
        
        n = len(groups)
        # Calculate products once for efficiency
        prods = [prod(g) for g in groups]
        
        for i in range(n):
            if prods[i] >= 512:
                continue
            for j in range(i + 1, n):
                if prods[j] >= 512:
                    continue
                new_prod = prods[i] * prods[j]
                if new_prod < 512:
                    # Calculate gain
                    gain = (1.0 / (512.0 - new_prod)) - (1.0 / (512.0 - prods[i])) - (1.0 / (512.0 - prods[j]))
                    if gain > best_gain:
                        best_gain = gain
                        best_i, best_j = i, j
        
        if best_gain > 1e-12:
            # Merge best_i and best_j
            new_group = groups[best_i] + groups[best_j]
            # Remove the two groups and add the merged one
            # To maintain index stability, remove the higher index first
            if best_j > best_i:
                groups.pop(best_j)
                groups.pop(best_i)
            else:
                groups.pop(best_i)
                groups.pop(best_j)
            groups.append(new_group)
        else:
            break
            
    # 4. Convert groups to final numbers
    result = [prod(g) for g in groups]
    
    # Optional: Verify pairwise coprime (should be true by construction)
    # for i in range(len(result)):
    #    for j in range(i+1, len(result)):
    #       if gcd(result[i], result[j]) != 1:
    #           return [] # Fallback
    
    return result
