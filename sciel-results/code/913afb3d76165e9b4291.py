"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
# EVOLVE-BLOCK-START
def construct():
    N = 512
    
    # 1. Find the number M <= N with the maximum number of divisors.
    # This provides a strong baseline (size 24 for N=512).
    best_M = 1
    max_divs = 0
    for m in range(1, N + 1):
        count = 0
        i = 1
        while i * i <= m:
            if m % i == 0:
                count += 1
                if i != m // i:
                    count += 1
            i += 1
        if count > max_divs:
            max_divs = count
            best_M = m
            
    # Generate divisors of best_M
    divisors = []
    i = 1
    while i * i <= best_M:
        if best_M % i == 0:
            divisors.append(i)
            if i != best_M // i:
                divisors.append(best_M // i)
        i += 1
        
    best_set = set(divisors)
    
    # 2. Try to improve using a greedy/local search on the conflict graph.
    # We check if we can add any number from 1..N not in best_set.
    # If we add x, we must ensure lcm(x, y) <= N for all y in best_set.
    # If adding x causes conflicts, we might remove some y's to add x and others.
    # Given time constraints, we do a simple check for single additions.
    
    for x in range(1, N + 1):
        if x in best_set:
            continue
        # Check if x is compatible with all elements in best_set
        compatible = True
        for y in best_set:
            # Optimization: if y is large, lcm is likely large
            # lcm(x, y) = x * y // gcd(x, y)
            if x * y // math.gcd(x, y) > N:
                compatible = False
                break
        if compatible:
            best_set.add(x)
            
    # 3. Try a simple local search: 
    # Remove one element, try to add multiple?
    # This is complex. We stick to the divisor set + single additions.
    
    return list(best_set)
# EVOLVE-BLOCK-END
