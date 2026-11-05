import itertools
import math

def construct():
    # Primes to consider. 
    # We need 15 primes with product <= 10^18.
    # First 15 primes product is ~6e17.
    # We can swap some for larger ones.
    # Pool of first 20 primes is sufficient for local search around the optimum.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    best_score = 0
    
    # Search for 15 primes
    for combo in itertools.combinations(primes, 15):
        n = 1
        # Early termination if product exceeds limit
        for p in combo:
            n *= p
            if n > 10**18:
                break
        if n > 10**18:
            continue
        
        # Generate divisors
        divs = [1]
        for p in combo:
            # Create new divisors by multiplying existing ones by p
            # This works because n is square-free
            divs += [d * p for d in divs]
        
        divs.sort()
        
        score = 0
        # Check adjacent pairs
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        
        if score > best_score:
            best_score = score
            best_n = n
            
    return best_n
