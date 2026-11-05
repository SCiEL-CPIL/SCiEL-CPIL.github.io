import time
import random
from itertools import product

def _count_nonzero(arr):
    return sum(1 for x in arr if x != 0)

def _square_nonzero_count_dense(coeffs, n):
    """Compute nonzero count in square of dense polynomial with n terms at exponents 0..n-1."""
    square = [0] * (2 * n - 1)
    for i in range(n):
        ci = coeffs[i]
        if ci == 0:
            continue
        for j in range(n):
            square[i + j] += ci * coeffs[j]
    return _count_nonzero(square)

def _square_nonzero_count_sparse(exponents, coeffs):
    """Compute nonzero count in square of sparse polynomial."""
    max_deg = exponents[-1] if exponents else 0
    square = [0] * (2 * max_deg + 1)
    n = len(exponents)
    for i in range(n):
        ei = exponents[i]
        ci = coeffs[i]
        if ci == 0:
            continue
        for j in range(n):
            ej = exponents[j]
            cj = coeffs[j]
            if cj == 0:
                continue
            square[ei + ej] += ci * cj
    return _count_nonzero(square)

def _build_full_coeff_list(exponents, coeffs):
    """Build full coefficient list in ascending degree."""
    if not exponents:
        return [0]
    max_deg = max(exponents)
    full = [0] * (max_deg + 1)
    for e, c in zip(exponents, coeffs):
        full[e] = c
    return full

def construct():
    start_time = time.time()
    timeout = 55.0

    best_count = float('inf')
    best_coeffs = [1] * 12

    # Phase 1: Brute force signs for 12 consecutive terms (0..11)
    for bits in range(1 << 12):
        if time.time() - start_time > timeout:
            break
        coeffs = [1 if (bits >> i) & 1 else -1 for i in range(12)]
        count = _square_nonzero_count_dense(coeffs, 12)
        if count < best_count:
            best_count = count
            best_coeffs = coeffs

    # Phase 2: Factor-based structured search
    # P(x) = A(x)*B(x)*C(x) where each factor has 2-3 terms
    # We search for factor structures where the product has exactly 12 terms
    # and the square has minimal support
    random.seed(42)
    
    # Try products of 3 two-term factors: (a + b*x^i)(c + d*x^j)(e + f*x^k)
    # This gives up to 8 terms in the product, so we need to be more careful
    # Instead, let's try: one 2-term factor times one 6-term factor
    # Or: two 3-term factors times one 2-term factor = 18 terms max, but many may cancel
    
    # Strategy: Search for P = Q * R where Q has 2 terms and R has 6 terms
    # Q = a + b*x^d, R has 6 terms
    # Then P has at most 12 terms (if no overlaps)
    
    # First, find good 6-term polynomials R with sparse squares
    # Then multiply by a 2-term factor Q
    
    # Phase 2a: Systematic search for 2-term * 6-term structures
    # Q = [1, 0, ..., 0, 1] at positions 0 and d (or with coefficients)
    # R = 6 terms at chosen exponents
    
    # Try Q with coefficient patterns
    for d in range(1, 20):
        if time.time() - start_time > timeout:
            break
        # Q = 1 + x^d or 1 - x^d
        for sign_q in [1, -1]:
            # Q coefficients: 1 at pos 0, sign_q at pos d
            # Now search for R with 6 terms such that (1 + sign_q*x^d)*R has 12 nonzero terms
            # and the square is sparse
            
            # R must have 6 terms; when multiplied by (1 + sign_q*x^d),
            # we get R + sign_q*x^d*R
            # For exactly 12 nonzero terms, we need no overlap between R and x^d*R
            # i.e., all exponents of R and all exponents of R+d must be distinct
            
            # Try R with exponents in [0, d-1] (so R and R+d don't overlap)
            # R has 6 terms from 6 positions in [0, d-1]
            if d < 6:
                continue
                
            # Choose 6 exponents from [0, d-1]
            # To keep it tractable, try consecutive or systematic patterns
            for r_start in range(max(0, d - 11), d - 5 + 1):
                if time.time() - start_time > timeout:
                    break
                # R exponents: r_start, r_start+1, ..., r_start+5
                r_exps = list(range(r_start, r_start + 6))
                
                # Search coefficients for R: each in {-2, -1, 1, 2}
                # That's 4^6 = 4096 combinations
                for rc in product([-2, -1, 1, 2], repeat=6):
                    if time.time() - start_time > timeout:
                        break
                    # Build P = (1 + sign_q*x^d) * R
                    # P exponents: r_exps and [e+d for e in r_exps]
                    p_exps = r_exps + [e + d for e in r_exps]
                    p_coeffs = list(rc) + [sign_q * c for c in rc]
                    
                    # Check exactly 12 nonzero terms
                    if _count_nonzero(p_coeffs) != 12:
                        continue
                    
                    # Check all coefficients in [-64, 64]
                    if any(abs(c) > 64 for c in p_coeffs):
                        continue
                    
                    count = _square_nonzero_count_sparse(p_exps, p_coeffs)
                    if count < best_count:
                        best_count = count
                        best_coeffs = _build_full_coeff_list(p_exps, p_coeffs)
                        if best_count == 12:
                            return best_coeffs

    # Phase 2b: Try Q with different coefficient values
    for d in range(1, 25):
        if time.time() - start_time > timeout:
            break
        for qa in [-2, -1, 1, 2]:
            for qb in [-2, -1, 1, 2]:
                if time.time() - start_time > timeout:
                    break
                # Q = qa + qb*x^d
                if d < 6:
                    continue
                for r_start in range(max(0, d - 11), d - 5 + 1):
                    if time.time() - start_time > timeout:
                        break
                    r_exps = list(range(r_start, r_start + 6))
                    
                    # Sample some coefficient patterns for R
                    for _ in range(50):
                        if time.time() - start_time > timeout:
                            break
                        rc = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
                        p_exps = r_exps + [e + d for e in r_exps]
                        p_coeffs = [qa * c for c in rc] + [qb * c for c in rc]
                        
                        if _count_nonzero(p_coeffs) != 12:
                            continue
                        if any(abs(c) > 64 for c in p_coeffs):
                            continue
                        
                        count = _square_nonzero_count_sparse(p_exps, p_coeffs)
                        if count < best_count:
                            best_count = count
                            best_coeffs = _build_full_coeff_list(p_exps, p_coeffs)
                            if best_count == 12:
                                return best_coeffs

    # Phase 3: Random search for coefficients in {-2, -1, 1, 2} with exponents 0..11
    while time.time() - start_time < timeout:
        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        count = _square_nonzero_count_dense(coeffs, 12)
        if count < best_count:
            best_count = count
            best_coeffs = coeffs
            if best_count == 12:
                return best_coeffs

    # Phase 4: Random search with random exponents and coefficients
    while time.time() - start_time < timeout:
        exponents = random.sample(range(20), 12)
        exponents.sort()
        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
        
        count = _square_nonzero_count_sparse(exponents, coeffs)
        if count < best_count:
            best_count = count
            best_coeffs = _build_full_coeff_list(exponents, coeffs)
            if best_count == 12:
                return best_coeffs

    # Phase 5: Local search / perturbation from best found
    if best_count > 12 and time.time() - start_time < timeout:
        # Try to improve by perturbing the best solution
        current = list(best_coeffs)
        current_max_deg = len(current) - 1
        
        for _ in range(2000):
            if time.time() - start_time > timeout:
                break
            candidate = list(current)
            # Randomly change 1-2 coefficients
            n_changes = random.randint(1, 2)
            for _ in range(n_changes):
                idx = random.randint(0, current_max_deg)
                candidate[idx] = random.choice([-2, -1, 0, 1, 2])
            
            # Check exactly 12 nonzero terms
            nonzero = [(i, c) for i, c in enumerate(candidate) if c != 0]
            if len(nonzero) != 12:
                continue
            if any(abs(c) > 64 for _, c in nonzero):
                continue
            
            exps = [i for i, _ in nonzero]
            coefs = [c for _, c in nonzero]
            count = _square_nonzero_count_sparse(exps, coefs)
            if count < best_count:
                best_count = count
                current = candidate
                best_coeffs = candidate
                if best_count == 12:
                    return best_coeffs

    return best_coeffs
