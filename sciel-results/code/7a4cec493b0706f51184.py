"""Bootstrap SEARCH program for sparse squares of 12-term integer polynomials."""
import numpy as np
import random
import sys
import time

# EVOLVE-BLOCK-START

_DEADLINE = time.time() + 85.0  # 5s margin before 90s kill

def _is_deadline():
    return time.time() >= _DEADLINE

def _square_count_fast(coeffs):
    """
    Compute the number of nonzero terms in the square of the polynomial
    defined by coeffs (list of ints, ascending degree).
    """
    if not coeffs:
        return 0
    arr = np.array(coeffs, dtype=np.int64)
    sq = np.convolve(arr, arr, mode='full')
    return int(np.count_nonzero(sq))

def _normalize(poly):
    """
    Validate and normalize the polynomial.
    Returns None if invalid.
    """
    if not poly:
        return None
    # Trim trailing zeros
    while len(poly) > 1 and poly[-1] == 0:
        poly = poly[:-1]
    
    if len(poly) > 41:
        return None
    if poly[-1] == 0:
        return None
        
    nz = sum(1 for c in poly if c != 0)
    if nz != 12:
        return None
        
    for c in poly:
        if not (-64 <= c <= 64):
            return None
            
    return poly

def _generate_factor_pair(rng):
    """
    Generate a random pair of 6-term polynomials (Q, R) with coefficients in [-64, 64].
    Returns (q_coeffs, r_coeffs) as lists.
    """
    def random_6_term_poly():
        degrees = sorted(rng.sample(range(0, 21), 6)) # Keep degrees small initially
        poly = [0] * 21
        for d in degrees:
            c = rng.randint(1, 64)
            if rng.random() < 0.5:
                c = -c
            poly[d] = c
        # Trim
        last_nz = max(d for d in degrees if poly[d] != 0)
        return poly[:last_nz+1]
    
    return random_6_term_poly(), random_6_term_poly()

def _local_search_factorized(q, r, rng, max_steps=2000):
    """
    Local search on the factorized representation.
    We modify Q or R coefficients/degrees and check the square count of P = Q*R.
    """
    q = list(q)
    r = list(r)
    
    def get_p_score():
        p = np.convolve(q, r, mode='full')
        # Check validity of P
        if len(p) > 41:
            return float('inf')
        # Check nonzero count of P
        nz_p = np.count_nonzero(p)
        if nz_p != 12:
            return float('inf')
        # Check coefficient bounds of P
        if np.any(np.abs(p) > 64):
            return float('inf')
        # Check final coeff nonzero
        if p[-1] == 0:
            return float('inf')
        # Check square count
        sq = np.convolve(p, p, mode='full')
        return int(np.count_nonzero(sq))

    current_score = get_p_score()
    if current_score == float('inf'):
        return None, float('inf')
    
    best_q = list(q)
    best_r = list(r)
    best_score = current_score
    
    for _ in range(max_steps):
        if _is_deadline():
            break
            
        improved = False
        
        # 1. Try changing a coefficient in Q
        if len(q) > 0:
            idx_q = rng.randrange(len(q))
            if q[idx_q] != 0:
                orig_q = q[idx_q]
                for _ in range(5):
                    new_val = rng.randint(-64, 64)
                    if new_val == 0 or new_val == orig_q:
                        continue
                    q[idx_q] = new_val
                    score = get_p_score()
                    if score < best_score:
                        best_score = score
                        best_q = list(q)
                        best_r = list(r)
                        improved = True
                    else:
                        q[idx_q] = orig_q
                if improved:
                    continue # Restart loop with new best state? Or just continue scanning?
                    # Let's restart the inner loop logic by breaking to top of outer loop
                    # Actually, standard hill climbing: accept and keep going.
                    # But since we changed q, indices might shift if we trim? 
                    # We don't trim inside loop for speed, but get_p_score handles it.
                    # However, if we set a coeff to 0, we have <6 terms in Q. 
                    # The problem requires P to have 12 terms, Q and R can have any number?
                    # No, the factorization is just a search heuristic. Q and R can have any number of terms 
                    # as long as their product P is valid.
                    # BUT, if Q has 5 terms and R has 6, P has at most 11 terms? No, 5+6-1=10 min, 10 max? 
                    # If Q has 5 terms and R has 6, P can have up to 10 terms? 
                    # Wait, if Q has 5 terms and R has 6, the product can have up to 5*6=30 terms, 
                    # but distinct degrees can be up to (max_deg_Q + max_deg_R).
                    # The constraint is on P: exactly 12 nonzero terms.
                    # So Q and R can have fewer than 6 terms, as long as P has 12.
                    # However, if we allow Q to lose terms, we might drift away from the "12 term" structure 
                    # if P becomes sparse in a bad way.
                    # Let's enforce that Q and R have at least 3 terms to keep the product "rich" enough 
                    # to potentially have 12 terms, but allow down to 1? 
                    # Actually, if Q has 1 term, P is monomial * R, so P has same sparsity as R. 
                    # If R has 6 terms, P has 6 terms. Invalid (need 12).
                    # So we need P to have 12 terms. 
                    # Let's just rely on get_p_score to reject invalid P.
                    pass
        
        # 2. Try changing a coefficient in R
        if len(r) > 0:
            idx_r = rng.randrange(len(r))
            if r[idx_r] != 0:
                orig_r = r[idx_r]
                for _ in range(5):
                    new_val = rng.randint(-64, 64)
                    if new_val == 0 or new_val == orig_r:
                        continue
                    r[idx_r] = new_val
                    score = get_p_score()
                    if score < best_score:
                        best_score = score
                        best_q = list(q)
                        best_r = list(r)
                        improved = True
                    else:
                        r[idx_r] = orig_r
        
        # 3. Try moving a degree in Q
        # Pick a nonzero term in Q, move it to a random empty degree
        nz_q = [i for i, c in enumerate(q) if c != 0]
        if nz_q:
            idx_q = rng.choice(nz_q)
            val_q = q[idx_q]
            q[idx_q] = 0
            # Find empty spot in Q
            empty_q = [i for i, c in enumerate(q) if c == 0]
            if empty_q:
                new_deg_q = rng.choice(empty_q)
                q[new_deg_q] = val_q
                score = get_p_score()
                if score < best_score:
                    best_score = score
                    best_q = list(q)
                    best_r = list(r)
                    improved = True
                else:
                    # Revert
                    q[new_deg_q] = 0
                    q[idx_q] = val_q
            else:
                q[idx_q] = val_q

        # 4. Try moving a degree in R
        nz_r = [i for i, c in enumerate(r) if c != 0]
        if nz_r:
            idx_r = rng.choice(nz_r)
            val_r = r[idx_r]
            r[idx_r] = 0
            empty_r = [i for i, c in enumerate(r) if c == 0]
            if empty_r:
                new_deg_r = rng.choice(empty_r)
                r[new_deg_r] = val_r
                score = get_p_score()
                if score < best_score:
                    best_score = score
                    best_q = list(q)
                    best_r = list(r)
                    improved = True
                else:
                    r[new_deg_r] = 0
                    r[idx_r] = val_r
            else:
                r[idx_r] = val_r

        if not improved:
            # Try a random restart of one factor?
            # Or break?
            # Let's try a random perturbation of one factor's coefficients
            if rng.random() < 0.1:
                # Randomize Q
                q[:] = [0] * len(q)
                degrees = sorted(rng.sample(range(0, 21), rng.randint(3, 6)))
                for d in degrees:
                    c = rng.randint(1, 64)
                    if rng.random() < 0.5: c = -c
                    q[d] = c
                score = get_p_score()
                if score < best_score:
                    best_score = score
                    best_q = list(q)
                    best_r = list(r)
                    improved = True

    # Reconstruct P from best_q and best_r
    p = np.convolve(best_q, best_r, mode='full')
    if len(p) > 41 or np.count_nonzero(p) != 12 or np.any(np.abs(p) > 64) or p[-1] == 0:
        return None, float('inf')
    
    return list(p), best_score

def _search():
    best_poly = None
    best_score = float('inf')
    
    rng = random.Random(42)
    
    # Phase 1: Factorized Search
    # We generate random Q, R with 6 terms each, then local search.
    # We also try structured Q, R like (1, x, x^2...) with specific coeffs.
    
    for restart in range(200):
        if _is_deadline():
            break
        
        # Generate random factors
        q, r = _generate_factor_pair(rng)
        
        # Local search
        p_candidate, score = _local_search_factorized(q, r, rng, max_steps=1000)
        
        if p_candidate is not None and score < best_score:
            best_score = score
            best_poly = p_candidate
            
            # If we find a very good one, we might stop early or continue
            if best_score <= 16:
                # Try to squeeze a bit more with more steps?
                # For now, just record and continue to see if we can beat it
                pass

    # Phase 2: Structured Factor Candidates
    # Try Q(x) = 1 + x + x^2 + x^3 + x^4 + x^5
    # Try R(x) = 1 - x + x^2 - x^3 + x^4 - x^5
    # P = Q*R
    # This often leads to cancellations.
    
    structured_qs = [
        [1, 1, 1, 1, 1, 1],
        [1, -1, 1, -1, 1, -1],
        [1, 2, 1, 2, 1, 2],
        [1, 1, -1, -1, 1, 1],
        [1, 0, 1, 0, 1, 0], # 3 terms, but let's see
    ]
    
    for sq in structured_qs:
        if _is_deadline():
            break
        for sr in structured_qs:
            if _is_deadline():
                break
            q = list(sq)
            r = list(sr)
            p_candidate, score = _local_search_factorized(q, r, rng, max_steps=500)
            if p_candidate is not None and score < best_score:
                best_score = score
                best_poly = p_candidate

    if best_poly is not None:
        return best_poly
        
    # Fallback to random search if factorized search failed completely
    # (This shouldn't happen often, but good for robustness)
    for _ in range(50):
        if _is_deadline():
            break
        poly = [0] * 41
        degrees = sorted(rng.sample(range(0, 41), 12))
        for d in degrees:
            c = rng.randint(1, 64)
            if rng.random() < 0.5: c = -c
            poly[d] = c
        last_nz = max(d for d in degrees if poly[d] != 0)
        p_candidate = poly[:last_nz+1]
        norm = _normalize(p_candidate)
        if norm:
            score = _square_count_fast(norm)
            if score < best_score:
                best_score = score
                best_poly = norm

    if best_poly is not None:
        return best_poly

    # Ultimate fallback
    fallback = [0] * 41
    for i in range(12):
        fallback[i] = 1
    return _normalize(fallback)

# EVOLVE-BLOCK-END

def construct():
    """Return coefficient list in ascending degree, length<=41, nonzero final
    coefficient, entries integers in [-64,64], exactly 12 nonzero terms.
    Minimize nonzero terms in the EXACT integer square."""
    result = _search()
    if result is None:
        # Ultimate fallback
        result = [1] * 12
        # Ensure valid
        result = _normalize(result)
        if result is None:
             # Last resort valid poly
             result = [0]*41
             for i in range(12):
                 result[i] = 1
             result = _normalize(result)
    return result
