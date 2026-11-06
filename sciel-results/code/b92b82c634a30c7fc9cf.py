import time
import numpy as np
import random
from itertools import product, combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Fallback: (1 + x + ... + x^11)^2 has 21 non-zero terms
    fallback = [1] * 12
    fallback_score = 21

    best_score = fallback_score
    best_candidate = fallback

    def compute_score_fast(exps, coeffs):
        """
        Fast computation of nonzero terms in P(x)^2.
        exps: list of 12 exponents (sorted)
        coeffs: list of 12 coefficients
        Returns count of nonzero terms, or inf if invalid.
        """
        if not exps or not coeffs:
            return float('inf')

        max_deg = max(exps)
        if max_deg > 40:
            return float('inf')
        if any(abs(c) > 64 for c in coeffs):
            return float('inf')
        if any(c == 0 for c in coeffs):
            return float('inf')
        if len(set(exps)) != 12:
            return float('inf')

        # Build dense array
        p = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in zip(exps, coeffs):
            p[e] = c

        q = np.convolve(p, p)
        count = int(np.count_nonzero(q))
        return count

    def _update_best(exps, coeffs, score):
        """Update the global best candidate if improved."""
        nonlocal best_score, best_candidate
        if score < best_score:
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            if len(dense) <= 41 and dense[-1] != 0:
                best_score = score
                best_candidate = dense

    def factor_product_score(factors):
        """
        Given 3 factors, each a list of (exp, coeff) pairs (4 each),
        compute the number of nonzero terms in P(x)^2 where P = F1*F2*F3.
        Uses the fact that P^2 = F1^2 * F2^2 * F3^2, and we can compute
        the support of the convolution efficiently.
        """
        # Each factor has 4 terms. F_i^2 has at most 10 terms (4*4 - overlaps).
        # P = F1*F2*F3, so P has 4*4*4 = 64 terms max, but many may cancel.
        # Actually, let's just build P directly and square it.
        
        # Build F1, F2, F3 as dense arrays
        max_exp = max(max(e for e, c in f) for f in factors)
        # P = F1*F2*F3, max degree = 3*max_exp
        # But we need max degree of P <= 40, so 3*max_exp <= 40 => max_exp <= 13
        
        if max_exp > 13:
            return float('inf')
        
        # Build each factor
        arrays = []
        for f in factors:
            arr = np.zeros(max_exp + 1, dtype=np.int64)
            for e, c in f:
                if 0 <= e <= max_exp and abs(c) <= 64 and c != 0:
                    arr[e] = c
                else:
                    return float('inf')
            arrays.append(arr)
        
        # P = F1 * F2 * F3
        p = np.convolve(np.convolve(arrays[0], arrays[1]), arrays[2])
        
        # Check constraints on P
        if len(p) > 41:
            return float('inf')
        if p[-1] == 0:
            # Trim trailing zeros
            while len(p) > 1 and p[-1] == 0:
                p = p[:-1]
            if p[-1] == 0:
                return float('inf')
        
        # Count nonzero terms in P
        nonzero_in_p = np.count_nonzero(p)
        if nonzero_in_p != 12:
            return float('inf')
        
        # Check coefficients
        if any(abs(c) > 64 for c in p):
            return float('inf')
        
        # Compute P^2
        q = np.convolve(p, p)
        return int(np.count_nonzero(q))

    def optimize_factor_structure():
        """
        Search over 3-factor decompositions.
        Each factor has 4 terms with exponents in [0, 13] (since 3*13=39 <= 40).
        We try to find factor structures where P = F1*F2*F3 has exactly 12
        nonzero terms and P^2 has minimal nonzero terms.
        """
        nonlocal best_score, best_candidate
        
        # Strategy: 
        # 1. Choose 3 sets of 4 exponents for each factor
        # 2. Choose coefficients for each factor
        # 3. Compute P = F1*F2*F3, check if it has exactly 12 nonzero terms
        # 4. If so, count nonzero terms in P^2
        
        # To make this tractable, we'll use a structured approach:
        # - Try small exponent ranges for each factor
        # - Use sign patterns that encourage cancellation
        
        coeff_choices = [-1, 1, -2, 2, -3, 3, -4, 4, -8, 8]
        
        # Precompute: for a factor with 4 terms, what's the max degree of F^2?
        # If factor has exponents up to e_max, F^2 has degree up to 2*e_max
        
        # Let's try structured factor exponents
        # Factor 1: low exponents, Factor 2: mid, Factor 3: high
        # But they multiply, so we need the total to be <= 40
        
        # Try: F1 exponents in [0,5], F2 in [0,5], F3 in [0,5]
        # Then P has degree up to 15, P^2 up to 30 - within bounds
        
        # Even better: try to find factors where P naturally has 12 terms
        # (1+x)(1+x^2)(1+x^3)(1+x^4) = 1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 + x^8 + x^9 + x^10 + x^11 + x^12
        # That's 13 terms! Close to 12.
        
        # Let's try (1+x)(1+x^2)(1+x^3)(1+x^4) with some signs flipped
        # to reduce to 12 terms
        
        # Generate candidate factor structures
        base_structures = []
        
        # Structure 1: Consecutive small exponents
        for s1 in range(0, 3):
            for s2 in range(0, 3):
                for s3 in range(0, 3):
                    f1 = [(s1 + i, 1) for i in range(4)]
                    f2 = [(s2 + i, 1) for i in range(4)]
                    f3 = [(s3 + i, 1) for i in range(4)]
                    base_structures.append((f1, f2, f3))
        
        # Structure 2: Gapped exponents
        for gap in [2, 3, 4]:
            for offset in range(0, 5):
                f1 = [(offset + i*gap, 1) for i in range(4)]
                f2 = [(offset + i*gap, 1) for i in range(4)]
                f3 = [(offset + i*gap, 1) for i in range(4)]
                if max(e for e, c in f3) <= 13:
                    base_structures.append((f1, f2, f3))
        
        # Structure 3: Mixed
        f1 = [(0, 1), (1, 1), (2, 1), (4, 1)]
        f2 = [(0, 1), (1, 1), (2, 1), (3, 1)]
        f3 = [(0, 1), (1, 1), (2, 1), (3, 1)]
        base_structures.append((f1, f2, f3))
        
        # For each structure, try different coefficient patterns
        for f1, f2, f3 in base_structures:
            if time.time() > deadline - 15:
                break
            
            # Try all sign patterns for the 12 coefficients
            # That's 2^12 = 4096 patterns - feasible
            for signs in product([-1, 1], repeat=12):
                if time.time() > deadline - 10:
                    break
                
                # Apply signs to factors
                sf1 = [(e, c * signs[i]) for i, (e, c) in enumerate(f1)]
                sf2 = [(e, c * signs[i+4]) for i, (e, c) in enumerate(f2)]
                sf3 = [(e, c * signs[i+8]) for i, (e, c) in enumerate(f3)]
                
                score = factor_product_score([sf1, sf2, sf3])
                if score < float('inf') and score < best_score:
                    # Reconstruct P
                    max_exp = max(max(e for e, c in f) for f in [sf1, sf2, sf3])
                    arrs = []
                    for f in [sf1, sf2, sf3]:
                        arr = np.zeros(max_exp + 1, dtype=np.int64)
                        for e, c in f:
                            arr[e] = c
                        arrs.append(arr)
                    p = np.convolve(np.convolve(arrs[0], arrs[1]), arrs[2])
                    # Trim
                    while len(p) > 1 and p[-1] == 0:
                        p = p[:-1]
                    if len(p) <= 41 and p[-1] != 0:
                        dense = list(p)
                        best_score = score
                        best_candidate = dense
        
        # Now do local search on the best factor structure found
        # Or: do a broader search with random factor structures
        if time.time() < deadline - 10:
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Random factor structure
                f1_exps = sorted(random.sample(range(0, 8), 4))
                f2_exps = sorted(random.sample(range(0, 8), 4))
                f3_exps = sorted(random.sample(range(0, 8), 4))
                
                f1 = [(e, random.choice([-1, 1])) for e in f1_exps]
                f2 = [(e, random.choice([-1, 1])) for e in f2_exps]
                f3 = [(e, random.choice([-1, 1])) for e in f3_exps]
                
                score = factor_product_score([f1, f2, f3])
                if score < float('inf') and score < best_score:
                    max_exp = max(max(e for e, c in f) for f in [f1, f2, f3])
                    arrs = []
                    for f in [f1, f2, f3]:
                        arr = np.zeros(max_exp + 1, dtype=np.int64)
                        for e, c in f:
                            arr[e] = c
                        arrs.append(arr)
                    p = np.convolve(np.convolve(arrs[0], arrs[1]), arrs[2])
                    while len(p) > 1 and p[-1] == 0:
                        p = p[:-1]
                    if len(p) <= 41 and p[-1] != 0:
                        dense = list(p)
                        best_score = score
                        best_candidate = dense

    def local_search_from_candidate():
        """
        Given the current best candidate, do local search by modifying
        the underlying factor structure or directly modifying P.
        """
        nonlocal best_score, best_candidate
        
        if time.time() > deadline - 5:
            return
        
        # Extract exponents and coefficients from best_candidate
        exps = [i for i, c in enumerate(best_candidate) if c != 0]
        coeffs = [c for c in best_candidate if c != 0]
        
        if len(exps) != 12:
            return
        
        # Try flipping signs of individual coefficients
        T = 5.0
        T_min = 0.5
        cooling = 0.99
        
        current_score = compute_score_fast(exps, coeffs)
        
        for it in range(3000):
            if time.time() > deadline - 3:
                break
            
            i = random.randint(0, 11)
            new_coeffs = coeffs[:]
            new_coeffs[i] = -new_coeffs[i]
            
            new_score = compute_score_fast(exps, new_coeffs)
            
            if new_score < current_score:
                coeffs = new_coeffs
                current_score = new_score
                _update_best(exps, coeffs, current_score)
            else:
                delta = new_score - current_score
                if T > T_min:
                    if random.random() < np.exp(-delta / T):
                        coeffs = new_coeffs
                        current_score = new_score
            
            T = max(T * cooling, T_min)

    # --- Main search loop ---

    random.seed(42)

    # Phase 1: Factor structure search
    optimize_factor_structure()

    # Phase 2: Local search on the result
    local_search_from_candidate()

    # Phase 3: If we haven't found something great, try direct random search
    if best_score > 15 and time.time() < deadline - 5:
        for _ in range(20):
            if time.time() > deadline - 3:
                break
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            score = compute_score_fast(exps, coeffs)
            if score < best_score:
                max_deg = max(exps)
                dense = [0] * (max_deg + 1)
                for e, c in zip(exps, coeffs):
                    dense[e] = c
                if len(dense) <= 41 and dense[-1] != 0:
                    best_score = score
                    best_candidate = dense

    return best_candidate

# EVOLVE-BLOCK-END
