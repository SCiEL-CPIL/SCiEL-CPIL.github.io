import numpy as np
import random
from itertools import product

def _square_nonzero_count(exponents, coeffs):
    """Compute number of nonzero terms in P^2 given sparse representation."""
    max_deg = 2 * max(exponents)
    poly = [0] * (max_deg + 1)
    for e, c in zip(exponents, coeffs):
        poly[e] = c
    sq = np.convolve(poly, poly)
    return int(np.count_nonzero(sq))

def construct():
    best_count = float('inf')
    best_exps = None
    best_coeffs = None
    
    def try_candidate(exps, coeffs):
        nonlocal best_count, best_exps, best_coeffs
        if len(exps) != 12 or len(coeffs) != 12:
            return
        if max(exps) > 40:
            return
        if any(c == 0 for c in coeffs):
            return
        if any(c < -64 or c > 64 for c in coeffs):
            return
        if len(set(exps)) != 12:
            return
        count = _square_nonzero_count(exps, coeffs)
        if count < best_count:
            best_count = count
            best_exps = list(exps)
            best_coeffs = list(coeffs)

    # Generate candidates from (1-x^k)^a (1+x^k)^b
    # We need exactly 12 nonzero terms in P.
    # If a+b+1 < 12, we can't get 12 terms from a single product of two binomials with step k.
    # But we can use step k and mix terms?
    # Or use (1-x^k)^a (1+x^k)^b where a,b are such that the product has 12 terms?
    # Product of (1-y)^a and (1+y)^b has a+b+1 terms.
    # So we need a+b+1 = 12 => a+b = 11.
    # Possible pairs (a,b) with a,b >= 0:
    # (0,11), (1,10), ..., (11,0).
    
    for a in range(12):
        b = 11 - a
        # P(y) = (1-y)^a (1+y)^b
        # Coeffs are computed by convolution of binomials
        poly_a = [(-1)**j * __import__('math').comb(a, j) for j in range(a+1)]
        poly_b = [__import__('math').comb(b, j) for j in range(b+1)]
        coeffs_y = np.convolve(poly_a, poly_b)
        
        # Check if exactly 12 nonzero coeffs
        if len(coeffs_y) != 12:
            continue
        if any(c == 0 for c in coeffs_y):
            continue
        if any(c < -64 or c > 64 for c in coeffs_y):
            continue
            
        # Exponents are 0, 1, ..., 11 times k
        # We need to fit in degree 40.
        # 11 * k <= 40 => k <= 3.
        for k in [1, 2, 3]:
            exps = [i * k for i in range(12)]
            try_candidate(exps, list(coeffs_y))

    # Also try (1-x^k)^a (1-x^{2k})^b ?
    # Or just random search with better heuristics?
    
    # Random search
    random.seed(12345)
    for _ in range(50000):
        exps = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.choice([-2,-1,1,2]) for _ in range(12)]
        try_candidate(exps, coeffs)
        
    # Local search on best
    if best_exps is not None:
        improved = True
        while improved:
            improved = False
            # Perturb coefficients
            for i in range(12):
                for new_c in [-64, -20, -5, -2, -1, 1, 2, 5, 20, 64]:
                    if new_c == 0:
                        continue
                    new_coeffs = list(best_coeffs)
                    new_coeffs[i] = new_c
                    try_candidate(best_exps, new_coeffs)
                    if best_count < float('inf'):
                        # Check if improved
                        if best_coeffs != new_coeffs:
                            improved = True
                            break
                if improved:
                    break
            
            # Perturb exponents
            if not improved:
                for i in range(12):
                    for delta in [-2, -1, 1, 2]:
                        new_exps = list(best_exps)
                        old_e = new_exps[i]
                        new_e = old_e + delta
                        if new_e < 0 or new_e > 40:
                            continue
                        if new_e in new_exps:
                            continue
                        new_exps[i] = new_e
                        new_exps.sort()
                        # Reassign coeffs?
                        # This is dangerous. We assume coeffs stay with their relative order?
                        # No, coeffs are tied to specific exponents.
                        # If we move an exponent, we should move its coeff.
                        # But try_candidate takes lists.
                        # We need to move coeff i to new position.
                        # Since we sort, the coeff order changes.
                        # This is complex. Skip exponent perturbation for now or do it carefully.
                        # Let's just try swapping exponents?
                        pass

    # Build result
    result = [0] * 41
    for e, c in zip(best_exps, best_coeffs):
        result[e] = c
    
    # Trim trailing zeros
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    if result[-1] == 0:
        result[-1] = best_coeffs[-1]
        
    return result
