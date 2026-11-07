"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    import math
    
    def get_divisors(n):
        divisors = []
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                divisors.append(i)
                if i != n // i:
                    divisors.append(n // i)
        return divisors

    best_max = float('inf')
    best_pair = None
    
    # Check k=1
    # n1 = 210 * t^2 > 210
    # Min t=2 -> n1=840
    # We can just include this in the general search if we treat it as k=1
    # But the loop below is for k=2.
    
    # Check k=2
    # n1 * n2 = 210 * t^2
    # We want to minimize max(n1, n2)
    # Iterate t
    for t in range(1, 1000):
        P = 210 * t * t
        
        # Optimization: if the smallest possible max for this t is already > best_max, break?
        # Smallest max is approx sqrt(P).
        # If sqrt(P) > best_max, and we can't get closer to sqrt(P) with factors > 210, 
        # we might break. But factors must be > 210.
        # If sqrt(P) < 210, no solution for this t (both factors > 210 implies P > 210^2).
        if P <= 210 * 211:
            continue
            
        divs = get_divisors(P)
        for d in divs:
            d2 = P // d
            if d == d2:
                continue
            if d > 210 and d2 > 210:
                m = max(d, d2)
                if m < best_max:
                    best_max = m
                    best_pair = (d, d2)
    
    if best_pair is not None:
        return [best_pair[0], best_pair[1]]
        
    # Fallback if no pair found (should not happen for t up to 1000)
    # Return k=1 solution
    return [840]

# EVOLVE-BLOCK-END
