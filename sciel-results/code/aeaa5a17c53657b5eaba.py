"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

def _count_nonzero(poly):
    return sum(1 for x in poly if x != 0)

def _square_poly(coeffs):
    # coeffs is list of (coef, exponent)
    # Result is dict {exponent: coef}
    res = {}
    n = len(coeffs)
    for i in range(n):
        c1, e1 = coeffs[i]
        for j in range(i, n):
            c2, e2 = coeffs[j]
            e_sum = e1 + e2
            if i == j:
                val = c1 * c1
            else:
                val = 2 * c1 * c2
            res[e_sum] = res.get(e_sum, 0) + val
    # Convert to list for counting
    # We only need to count non-zeros
    return sum(1 for v in res.values() if v != 0)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_score = 1000  # Max possible terms in square of 12 terms is 22 (if no collisions) or more if gaps? 
    # Actually max terms in square of 12 terms is 22 if exponents are distinct and sorted? 
    # (e1+e1, e1+e2, ..., e11+e11). 12 diagonal, 11*12/2 off-diagonal pairs? 
    # No, number of distinct sums. Max is 22 if exponents are super-increasing? 
    # But we want MIN.
    
    best_coeffs = None
    
    # Strategy 1: Arithmetic progression exponents, small integer coefficients
    # Exponents: 0..11
    # Coefficients: -1, 1
    # This is 2^12 = 4096 combinations. Very fast.
    
    # Strategy 2: Arithmetic progression exponents, coefficients in {-2, -1, 1, 2}
    # 4^12 = 16M. Might be too slow in Python if not optimized.
    # Let's stick to {-1, 1} first, then expand if time permits.
    
    # Strategy 3: Symmetric exponents around 0, e.g., -5..5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10?
    # Let's try different exponent sets.
    
    exponent_sets = [
        list(range(12)),
        list(range(-6, 6)),
        list(range(0, 24, 2)), # Even exponents
    ]
    
    # Search over sign patterns for each exponent set
    # We can use numpy for speed if available, but stdlib is safer for portability in this context?
    # NumPy is available.
    
    try:
        import numpy as np
        use_numpy = True
    except ImportError:
        use_numpy = False
        
    for exps in exponent_sets:
        if time.time() > deadline - 5.0:
            break
            
        # Generate all sign combinations
        # 2^12 = 4096
        for mask in range(4096):
            if time.time() > deadline - 5.0:
                break
                
            coeffs = []
            for i in range(12):
                if (mask >> i) & 1:
                    coeffs.append((1, exps[i]))
                else:
                    coeffs.append((-1, exps[i]))
                    
            # Calculate square non-zeros
            if use_numpy:
                # Construct polynomial array
                # Max degree is max(exps)*2
                max_deg = max(exps) * 2
                min_deg = min(exps) * 2
                # Shift to positive indices
                poly = np.zeros(max_deg - min_deg + 1, dtype=int)
                for c, e in coeffs:
                    poly[e - min_deg] = c
                sq = np.convolve(poly, poly)
                nz = np.count_nonzero(sq)
            else:
                nz = _square_poly(coeffs)
                
            if nz < best_score:
                best_score = nz
                best_coeffs = [c for c, e in coeffs]
                # Store exponents too for reconstruction? 
                # The return value is just the coefficient list.
                # But the evaluator needs to know the exponents?
                # "Return coefficient list in ascending degree"
                # This implies the list indices ARE the degrees.
                # So if I return [1, -1, 1...], it assumes degrees 0, 1, 2...
                # My search uses specific exponents.
                # I must return a list where index = degree.
                
    # If we found a good one with specific exponents, we must construct the full list.
    # But the loop above only kept the coefficients, not the mapping to the final list.
    # Let's refine: Keep the best (score, coeffs, exps).
    
    best_score = 1000
    best_result = None
    
    for exps in exponent_sets:
        if time.time() > deadline - 5.0:
            break
            
        for mask in range(4096):
            if time.time() > deadline - 5.0:
                break
                
            coeffs = []
            for i in range(12):
                if (mask >> i) & 1:
                    coeffs.append((1, exps[i]))
                else:
                    coeffs.append((-1, exps[i]))
                    
            if use_numpy:
                max_deg = max(exps) * 2
                min_deg = min(exps) * 2
                poly = np.zeros(max_deg - min_deg + 1, dtype=int)
                for c, e in coeffs:
                    poly[e - min_deg] = c
                sq = np.convolve(poly, poly)
                nz = np.count_nonzero(sq)
            else:
                nz = _square_poly(coeffs)
                
            if nz < best_score:
                best_score = nz
                # Construct the full coefficient list
                max_deg = max(exps)
                result = [0] * (max_deg + 1)
                for c, e in coeffs:
                    result[e] = c
                best_result = result
                
    if best_result is None:
        # Fallback
        return [1] * 12
        
    return best_result

# EVOLVE-BLOCK-END
