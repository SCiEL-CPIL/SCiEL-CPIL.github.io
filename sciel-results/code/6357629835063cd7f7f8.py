import numpy as np
from itertools import product

def count_nonzero(lst):
    return sum(1 for x in lst if x != 0)

def poly_square(coeffs):
    """Compute exact square of polynomial given as coeff list (ascending degree)."""
    n = len(coeffs)
    result = [0] * (2*n - 1)
    for i in range(n):
        if coeffs[i] == 0:
            continue
        for j in range(i, n):
            if coeffs[j] == 0:
                continue
            val = coeffs[i] * coeffs[j]
            if i == j:
                result[i+j] += val
            else:
                result[i+j] += 2 * val
    return result

def construct():
    # Try several candidate polynomials and return the one with fewest nonzero terms in square
    candidates = []
    
    # Candidate 1: (1+x+x^2+x^3)(1+x^4+x^8)
    p1 = [1,1,1,1,0,1,0,0,1]  # exponents 0,1,2,3,4,8... wait that's only 6 terms
    # Actually: (1+x+x^2+x^3) * (1+x^4+x^8) = 1+x+x^2+x^3 + x^4+x^5+x^6+x^7 + x^8+x^9+x^10+x^11
    p1 = [1,1,1,1,1,1,1,1,1,1,1,1]  # all 1s from 0 to 11
    candidates.append(p1)
    
    # Candidate 2: Try with some -1 coefficients to induce cancellation
    # P(x) = 1 - x + x^2 - x^3 + x^4 - x^5 + x^6 - x^7 + x^8 - x^9 + x^10 - x^11
    p2 = [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1]
    candidates.append(p2)
    
    # Candidate 3: Grouped
    # Try to find via small search
    best = None
    best_count = float('inf')
    
    for cand in candidates:
        sq = poly_square(cand)
        cnt = count_nonzero(sq)
        if cnt < best_count:
            best_count = cnt
            best = cand
    
    # Do a small random search
    import random
    random.seed(42)
    for _ in range(5000):
        # Random 12-term polynomial
        exponents = sorted(random.sample(range(41), 12))
        coeffs = [0] * 41
        for e in exponents:
            coeffs[e] = random.randint(-64, 64)
            if coeffs[e] == 0:
                coeffs[e] = 1
        sq = poly_square(coeffs)
        cnt = count_nonzero(sq)
        if cnt < best_count:
            best_count = cnt
            best = coeffs[:]
    
    # Trim to nonzero final coefficient
    while best and best[-1] == 0:
        best.pop()
    
    return best
