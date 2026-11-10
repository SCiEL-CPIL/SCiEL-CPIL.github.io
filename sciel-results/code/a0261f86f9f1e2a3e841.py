import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    best_result = [1] * 12  # fallback
    best_nonzero = 144  # worst case
    
    def poly_multiply(A, B):
        """Multiply two sparse polynomials (list of (coeff, exponent) pairs).
        Returns list of (coeff, exponent) for nonzero terms, sorted by exponent."""
        result = defaultdict(int)
        for ca, ea in A:
            for cb, eb in B:
                result[ea + eb] += ca * cb
        # Filter zeros and sort
        terms = [(e, c) for e, c in result.items() if c != 0]
        terms.sort()
        return [(c, e) for e, c in terms]
    
    def count_nonzero_square_from_terms(terms):
        """Count nonzero terms in square of polynomial given as list of (coeff, exponent)."""
        result = defaultdict(int)
        n = len(terms)
        for i in range(n):
            ci, ei = terms[i]
            for j in range(i, n):
                cj, ej = terms[j]
                d = ei + ej
                w = 1 if i == j else 2
                result[d] += w * ci * cj
        return sum(1 for v in result.values() if v != 0)
    
    def terms_to_coeff_list(terms):
        """Convert list of (coeff, exponent) to coefficient list in ascending degree."""
        if not terms:
            return [1]
        max_deg = max(e for _, e in terms)
        coeffs = [0] * (max_deg + 1)
        for c, e in terms:
            coeffs[e] = c
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs
    
    def update_best_from_terms(terms):
        """Update best if terms give fewer nonzero square terms."""
        nonlocal best_nonzero, best_result
        if len(terms) != 12:
            return
        if any(c == 0 for c, e in terms):
            return
        if any(c < -64 or c > 64 for c, e in terms):
            return
        if any(e < 0 for _, e in terms):
            return
        max_deg = max(e for _, e in terms)
        if 2 * max_deg + 1 > 41:
            return
        nz = count_nonzero_square_from_terms(terms)
        if nz < best_nonzero:
            best_nonzero = nz
            best_result = terms_to_coeff_list(terms)
    
    def build_degree_map(exponents):
        """Precompute degree -> list of (i, j, weight) for square terms"""
        n = len(exponents)
        dm = defaultdict(list)
        for i in range(n):
            for j in range(i, n):
                d = exponents[i] + exponents[j]
                w = 1 if i == j else 2
                dm[d].append((i, j, w))
        return dict(dm)
    
    def fast_square_nonzero(coeffs, exponents, degree_map):
        """Fast count using precomputed degree_map"""
        for d, pairs in degree_map.items():
            val = 0
            for (i, j, w) in pairs:
                val += w * coeffs[i] * coeffs[j]
            if val != 0:
                pass
        sq = {}
        for d, pairs in degree_map.items():
            val = 0
            for (i, j, w) in pairs:
                val += w * coeffs[i] * coeffs[j]
            sq[d] = val
        return sum(1 for v in sq.values() if v != 0)
    
    def is_valid_result(coeffs, exponents):
        if len(coeffs) != 12:
            return False
        if any(c == 0 for c in coeffs):
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        if any(e < 0 for e in exponents):
            return False
        max_deg = max(exponents)
        if 2 * max_deg + 1 > 41:
            return False
        return True
    
    def coeffs_from_state(state):
        """Convert state to coefficient list in ascending degree, length <= 41"""
        coeffs = [0] * (2 * max(state[1]) + 1)
        for c, e in zip(state[0], state[1]):
            coeffs[e] = c
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs
    
    def update_best(coeffs, exponents, nz):
        nonlocal best_nonzero, best_result
        if nz < best_nonzero:
            best_nonzero = nz
            result = coeffs_from_state((coeffs, exponents))
            if len(result) <= 41 and result[-1] != 0:
                best_result = result
    
    # FACTORIZATION STRATEGY: P = A * B where |A| * |B| = 12
    # Key insight: (A*B)^2 = A^2 * B^2, so sparsity of A^2 and B^2 
    # constrains sparsity of P^2. We search over factor pairs systematically.
    
    def enumerate_sparse_polys(num_terms, max_deg, max_coeff=64):
        """Generate candidate sparse polynomials with given number of terms."""
        candidates = []
        if num_terms == 1:
            for c in range(-max_coeff, max_coeff + 1):
                if c != 0:
                    for e in range(max_deg + 1):
                        candidates.append([(c, e)])
            return candidates
        if num_terms == 2:
            for c1 in range(-max_coeff, max_coeff + 1, 2):  # step 2 for speed
                if c1 == 0:
                    continue
                for c2 in range(-max_coeff, max_coeff + 1, 2):
                    if c2 == 0:
                        continue
                    for e1 in range(max_deg + 1):
                        for e2 in range(e1 + 1, max_deg + 1):
                            candidates.append([(c1, e1), (c2, e2)])
            return candidates
        return candidates  # For >2 terms, we'll use random generation
    
    def random_sparse_poly(rng, num_terms, max_deg, max_coeff=8):
        """Generate a random sparse polynomial."""
        exponents = sorted(rng.sample(range(max_deg + 1), num_terms))
        coeffs = []
        for _ in range(num_terms):
            mag = rng.choice([1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8])
            sign = 1 if rng.random() < 0.5 else -1
            coeffs.append(sign * mag)
        return list(zip(coeffs, exponents))
    
    def evaluate_factor_pair(A, B):
        """Multiply A and B, check if result has exactly 12 terms, evaluate square sparsity."""
        P = poly_multiply(A, B)
        if len(P) != 12:
            return None
        if any(c == 0 for c, e in P):
            return None
        if any(c < -64 or c > 64 for c, e in P):
            return None
        max_deg = max(e for _, e in P)
        if 2 * max_deg + 1 > 41:
            return None
        nz = count_nonzero_square_from_terms(P)
        return (nz, P)
    
    # Factor pair types: (3,4), (4,3), (2,6), (6,2), (1,12), (12,1)
    # We focus on (3,4) and (2,6) as they give the best balance
    
    rng = random.Random(42)
    
    # Strategy 1: Systematic search over 3-term and 4-term factors
    # For 3-term factors: choose 3 exponents and 3 coefficients
    # For 4-term factors: choose 4 exponents and 4 coefficients
    
    # Pre-generate candidate factor pools
    max_deg_A = 10  # max exponent for 3-term factor
    max_deg_B = 10  # max exponent for 4-term factor
    
    # Generate diverse 3-term factors
    factors_3 = []
    for _ in range(500):
        if time.time() > deadline:
            break
        A = random_sparse_poly(rng, 3, max_deg_A, max_coeff=6)
        factors_3.append(A)
    
    # Generate diverse 4-term factors
    factors_4 = []
    for _ in range(500):
        if time.time() > deadline:
            break
        B = random_sparse_poly(rng, 4, max_deg_B, max_coeff=6)
        factors_4.append(B)
    
    # Evaluate all pairs (with time limit)
    for A in factors_3:
        if time.time() > deadline:
            break
        for B in factors_4:
            if time.time() > deadline:
                break
            result = evaluate_factor_pair(A, B)
            if result is not None:
                nz, P = result
                if nz < best_nonzero:
                    best_nonzero = nz
                    best_result = terms_to_coeff_list(P)
    
    # Strategy 2: 2-term and 6-term factors
    max_deg_A2 = 15
    max_deg_B2 = 15
    
    factors_2 = []
    for _ in range(200):
        if time.time() > deadline:
            break
        A = random_sparse_poly(rng, 2, max_deg_A2, max_coeff=8)
        factors_2.append(A)
    
    factors_6 = []
    for _ in range(200):
        if time.time() > deadline:
            break
        B = random_sparse_poly(rng, 6, max_deg_B2, max_coeff=5)
        factors_6.append(B)
    
    for A in factors_2:
        if time.time() > deadline:
            break
        for B in factors_6:
            if time.time() > deadline:
                break
            result = evaluate_factor_pair(A, B)
            if result is not None:
                nz, P = result
                if nz < best_nonzero:
                    best_nonzero = nz
                    best_result = terms_to_coeff_list(P)
    
    # Strategy 3: Structured factor patterns with known good properties
    # Use factors with alternating signs to promote cancellation
    
    structured_3 = []
    for e1 in range(0, 8):
        for e2 in range(e1 + 1, 12):
            for e3 in range(e2 + 1, 16):
                for c1 in [1, -1, 2, -2]:
                    for c2 in [1, -1, 2, -2]:
                        for c3 in [1, -1, 2, -2]:
                            structured_3.append([(c1, e1), (c2, e2), (c3, e3)])
    
    structured_4 = []
    for e1 in range(0, 6):
        for e2 in range(e1 + 1, 10):
            for e3 in range(e2 + 1, 14):
                for e4 in range(e3 + 1, 18):
                    for c1 in [1, -1]:
                        for c2 in [1, -1]:
                            for c3 in [1, -1]:
                                for c4 in [1, -1]:
                                    structured_4.append([(c1, e1), (c2, e2), (c3, e3), (c4, e4)])
    
    # Sample from structured pools to avoid O(n^2) explosion
    rng2 = random.Random(99)
    sample_3 = rng2.sample(structured_3, min(200, len(structured_3)))
    sample_4 = rng2.sample(structured_4, min(200, len(structured_4)))
    
    for A in sample_3:
        if time.time() > deadline:
            break
        for B in sample_4:
            if time.time() > deadline:
                break
            result = evaluate_factor_pair(A, B)
            if result is not None:
                nz, P = result
                if nz < best_nonzero:
                    best_nonzero = nz
                    best_result = terms_to_coeff_list(P)
    
    # Strategy 4: Local search on factor space
    # Start from best found, perturb factors and re-evaluate
    if time.time() < deadline:
        # Get current best as terms
        nz_indices = [i for i, c in enumerate(best_result) if c != 0]
        if len(nz_indices) == 12:
            best_terms = [(best_result[i], i) for i in nz_indices]
            
            # Try to factor the best polynomial
            # For now, do random factor perturbation
            for _ in range(200):
                if time.time() > deadline:
                    break
                
                # Generate random factors near structured patterns
                A = random_sparse_poly(rng, 3, 12, max_coeff=6)
                B = random_sparse_poly(rng, 4, 12, max_coeff=6)
                
                result = evaluate_factor_pair(A, B)
                if result is not None:
                    nz, P = result
                    if nz < best_nonzero:
                        best_nonzero = nz
                        best_result = terms_to_coeff_list(P)
    
    # Strategy 5: Direct search with factor-aware local optimization
    if time.time() < deadline:
        # Use the factorization to guide local search
        # For each factor pair found, do local search on coefficients
        for A in factors_3[:50]:
            if time.time() > deadline:
                break
            for B in factors_4[:50]:
                if time.time() > deadline:
                    break
                result = evaluate_factor_pair(A, B)
                if result is None:
                    continue
                nz, P = result
                if nz >= best_nonzero:
                    continue
                
                # Local search: perturb coefficients of A and B
                for _ in range(30):
                    if time.time() > deadline:
                        break
                    # Perturb A
                    A_new = [(c + rng.choice([-1, 1]), e) for c, e in A]
                    A_new = [(c, e) for c, e in A_new if c != 0 and -64 <= c <= 64]
                    if len(A_new) != 3:
                        continue
                    
                    result2 = evaluate_factor_pair(A_new, B)
                    if result2 is not None:
                        nz2, P2 = result2
                        if nz2 < nz:
                            nz, P = nz2, P2
                            A = A_new
                    
                    # Perturb B
                    B_new = [(c + rng.choice([-1, 1]), e) for c, e in B]
                    B_new = [(c, e) for c, e in B_new if c != 0 and -64 <= c <= 64]
                    if len(B_new) != 4:
                        continue
                    
                    result3 = evaluate_factor_pair(A, B_new)
                    if result3 is not None:
                        nz3, P3 = result3
                        if nz3 < nz:
                            nz, P = nz3, P3
                            B = B_new
                
                if nz < best_nonzero:
                    best_nonzero = nz
                    best_result = terms_to_coeff_list(P)
    
    # Strategy 6: Classic exponent patterns (fallback)
    if time.time() < deadline:
        exponent_patterns = [
            list(range(12)),
            list(range(0, 24, 2)),
            [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
            [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        ]
        
        coeff_sets = [
            [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
            [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
            [2, -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2],
            [1, 2, -1, -2, 1, 2, -1, -2, 1, 2, -1, -2],
        ]
        
        for exponents in exponent_patterns:
            if time.time() > deadline:
                break
            dm = build_degree_map(exponents)
            for coeffs in coeff_sets:
                if not is_valid_result(coeffs, exponents):
                    continue
                nz = fast_square_nonzero(coeffs, exponents, dm)
                update_best(coeffs, exponents, nz)
    
    # Final local search on best result
    if time.time() < deadline:
        nz_indices = [i for i, c in enumerate(best_result) if c != 0]
        if len(nz_indices) == 12:
            best_exponents = nz_indices
            best_coeffs = [best_result[i] for i in nz_indices]
            dm = build_degree_map(best_exponents)
            
            current_coeffs = best_coeffs[:]
            current_nz = fast_square_nonzero(current_coeffs, best_exponents, dm)
            improved = True
            for _ in range(200):
                if time.time() > deadline:
                    break
                if not improved:
                    break
                improved = False
                best_flip = None
                best_nz = current_nz
                for flip_idx in range(12):
                    new_coeff = -current_coeffs[flip_idx]
                    nz = 0
                    for d, pairs in dm.items():
                        val = 0
                        for (i, j, w) in pairs:
                            ci = new_coeff if i == flip_idx else current_coeffs[i]
                            cj = new_coeff if j == flip_idx else current_coeffs[j]
                            val += w * ci * cj
                        if val != 0:
                            nz += 1
                    if nz < best_nz:
                        best_nz = nz
                        best_flip = flip_idx
                if best_flip is not None:
                    current_coeffs[best_flip] = -current_coeffs[best_flip]
                    current_nz = best_nz
                    improved = True
            
            if is_valid_result(current_coeffs, best_exponents):
                update_best(current_coeffs, best_exponents, current_nz)
    
    # Ensure result is valid
    if len(best_result) > 41 or best_result[-1] == 0:
        best_result = [1] * 12
    nz_count = sum(1 for c in best_result if c != 0)
    if nz_count != 12:
        best_result = [1] * 12
    if any(c < -64 or c > 64 for c in best_result):
        best_result = [1] * 12
    
    return best_result
