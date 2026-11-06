import time
import random
import numpy as np
from itertools import product as iproduct

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    best_score = float('inf')
    best_candidate = None

    def square_nonzero_count(pairs):
        """pairs: list of (exp, coeff) tuples. Returns (nonzero_count, valid)."""
        if len(pairs) != 12:
            return float('inf'), False
        max_deg = max(e for e, c in pairs)
        if max_deg > 40:
            return float('inf'), False
        p = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in pairs:
            if c == 0:
                return float('inf'), False
            if abs(c) > 64:
                return float('inf'), False
            p[e] = c
        square = np.convolve(p, p)
        return int(np.count_nonzero(square)), True

    def update_best(pairs, score):
        nonlocal best_score, best_candidate
        if score < best_score:
            best_score = score
            max_deg = max(e for e, c in pairs)
            dense = [0] * (max_deg + 1)
            for e, c in pairs:
                dense[e] = int(c)
            if len(dense) <= 41 and dense[-1] != 0:
                best_candidate = dense

    def product_of_three_four_term(exps1, exps2, exps3, coeffs1, coeffs2, coeffs3):
        """Build the 12-term polynomial as product of three 4-term factors."""
        p1 = np.zeros(41, dtype=np.int64)
        p2 = np.zeros(41, dtype=np.int64)
        p3 = np.zeros(41, dtype=np.int64)
        for e, c in zip(exps1, coeffs1):
            p1[e] = c
        for e, c in zip(exps2, coeffs2):
            p2[e] = c
        for e, c in zip(exps3, coeffs3):
            p3[e] = c
        prod = np.convolve(np.convolve(p1, p2), p3)
        pairs = [(i, int(c)) for i, c in enumerate(prod) if c != 0]
        return pairs

    def compute_square_structure(exps1, exps2, exps3):
        """
        Precompute the exponent mapping for the square of a 12-term polynomial
        formed as product of three 4-term factors with disjoint supports.
        
        Returns:
          - exp_to_index: dict mapping exponent -> list of (i,j,k) indices where
            e1[i]+e2[j]+e3[k] = exponent
          - exps_sorted: sorted list of all 12 exponents
        """
        exps_all = sorted(exps1 + exps2 + exps3)
        exp_set = set(exps_all)
        
        # Build mapping from exponent to list of (i,j,k)
        exp_to_index = {}
        for i, e1 in enumerate(exps1):
            for j, e2 in enumerate(exps2):
                for k, e3 in enumerate(exps3):
                    total = e1 + e2 + e3
                    if total not in exp_to_index:
                        exp_to_index[total] = []
                    exp_to_index[total].append((i, j, k))
        
        return exp_to_index, exps_all

    def optimize_product_structure_v2(exps1, exps2, exps3):
        """
        Optimize coefficients for a fixed product-of-three structure.
        Uses precomputed exponent mapping to efficiently evaluate sign patterns.
        """
        nonlocal best_score, best_candidate
        current_best = float('inf')
        current_best_signs = None
        
        # Precompute the structure
        exp_to_index, exps_all = compute_square_structure(exps1, exps2, exps3)
        
        # Check if all 12 exponents are present
        if len(exps_all) != 12 or len(exp_to_index) != 12:
            return float('inf'), None
        
        # For each sign pattern of the 12 base coefficients, compute the square
        # The coefficient at each exponent in P is the sum of products c1[i]*c2[j]*c3[k]
        # over all (i,j,k) with e1[i]+e2[j]+e3[k] = that exponent
        
        # Since we're only trying signs (±1), the coefficient at each exponent is
        # an integer between -4 and 4 (at most 4 terms can contribute to each exponent
        # in the worst case, but typically fewer)
        
        # Try all 2^12 = 4096 sign patterns
        all_signs = list(iproduct([-1, 1], repeat=12))
        
        for signs in all_signs:
            if time.time() > deadline - 5:
                break
            
            c1 = list(signs[0:4])
            c2 = list(signs[4:8])
            c3 = list(signs[8:12])
            
            # Compute the 12-term polynomial coefficients
            coeff_map = {}
            for exp, indices in exp_to_index.items():
                total = 0
                for (i, j, k) in indices:
                    total += c1[i] * c2[j] * c3[k]
                if total != 0:
                    coeff_map[exp] = total
            
            # Check validity
            if len(coeff_map) != 12:
                continue
            
            # Build pairs
            pairs = [(e, c) for e, c in coeff_map.items()]
            
            # Compute square nonzero count
            max_deg = max(e for e, c in pairs)
            if max_deg > 40:
                continue
            
            p = np.zeros(max_deg + 1, dtype=np.int64)
            for e, c in pairs:
                if abs(c) > 64:
                    break
                p[e] = c
            else:
                square = np.convolve(p, p)
                score = int(np.count_nonzero(square))
                if score < current_best:
                    current_best = score
                    current_best_signs = signs[:]
        
        if current_best_signs is not None:
            c1 = list(current_best_signs[0:4])
            c2 = list(current_best_signs[4:8])
            c3 = list(current_best_signs[8:12])
            pairs = product_of_three_four_term(exps1, exps2, exps3, c1, c2, c3)
            if len(pairs) == 12:
                update_best(pairs, current_best)
        
        return current_best, current_best_signs

    def optimize_exponents_for_product_v2(max_restarts=50):
        """
        Optimize exponent choices for the three 4-term factors.
        Uses structured exponent generation to encourage good cancellation.
        """
        nonlocal best_score, best_candidate
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Strategy: choose exponents such that the 12-term polynomial has
            # structure that encourages cancellation in the square.
            
            # Approach 1: Arithmetic progression base
            if restart % 3 == 0:
                base = random.randint(0, 5)
                step = random.choice([1, 2, 3, 4, 5, 6, 8])
                exps_all = [base + i * step for i in range(12)]
                if max(exps_all) > 40:
                    continue
                # Split into three groups of 4
                # Try different split patterns
                splits = [
                    ([0,1,2,3], [4,5,6,7], [8,9,10,11]),
                    ([0,4,8,2], [1,5,9,3], [6,10,11,7]),
                    ([0,3,6,9], [1,4,7,10], [2,5,8,11]),
                ]
                for s1, s2, s3 in splits:
                    if time.time() > deadline - 5:
                        break
                    exps1 = [exps_all[i] for i in s1]
                    exps2 = [exps_all[i] for i in s2]
                    exps3 = [exps_all[i] for i in s3]
                    _, _ = optimize_product_structure_v2(exps1, exps2, exps3)
            
            # Approach 2: Random with structure
            elif restart % 3 == 1:
                # Choose 12 exponents with some regularity
                base = random.randint(0, 3)
                offsets = sorted(random.sample(range(0, 40 - base), 12))
                exps_all = [base + o for o in offsets]
                exps1 = exps_all[0:4]
                exps2 = exps_all[4:8]
                exps3 = exps_all[8:12]
                _, _ = optimize_product_structure_v2(exps1, exps2, exps3)
            
            # Approach 3: Pure random
            else:
                exps_all = sorted(random.sample(range(0, 41), 12))
                exps1 = exps_all[0:4]
                exps2 = exps_all[4:8]
                exps3 = exps_all[8:12]
                _, _ = optimize_product_structure_v2(exps1, exps2, exps3)

    def local_search_on_pairs(pairs, max_iters=3000):
        """Local search on a 12-term polynomial."""
        current_pairs = pairs[:]
        current_score, valid = square_nonzero_count(current_pairs)
        if not valid:
            return
        update_best(current_pairs, current_score)
        
        temp = 1.0
        cooling = 0.9995
        
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            new_pairs = current_pairs[:]
            move_type = random.random()
            
            if move_type < 0.6:
                idx = random.randint(0, 11)
                old_exp, old_coeff = new_pairs[idx]
                new_coeff = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                if new_coeff == old_coeff:
                    continue
                new_pairs[idx] = (old_exp, new_coeff)
            else:
                idx = random.randint(0, 11)
                old_exp, old_coeff = new_pairs[idx]
                delta = random.choice([-3, -2, -1, 1, 2, 3])
                new_exp = old_exp + delta
                if new_exp < 0 or new_exp > 40:
                    continue
                other_exps = [e for j, (e, c) in enumerate(new_pairs) if j != idx]
                if new_exp in other_exps:
                    continue
                new_pairs[idx] = (new_exp, old_coeff)
                
            new_score, valid = square_nonzero_count(new_pairs)
            if not valid:
                continue
            
            if new_score < current_score:
                current_pairs = new_pairs
                current_score = new_score
                update_best(current_pairs, current_score)
            else:
                delta_e = new_score - current_score
                if random.random() < np.exp(-delta_e / temp):
                    current_pairs = new_pairs
                    current_score = new_score
            
            temp *= cooling

    # --- Phase 1: Product-of-three structure search with improved optimization ---
    optimize_exponents_for_product_v2(max_restarts=80)

    # --- Phase 2: Structured initial candidates + local search ---
    candidates = []
    
    # Arithmetic progressions
    for start in range(0, 10):
        for step in range(1, 5):
            exps = [start + i * step for i in range(12)]
            if max(exps) <= 40:
                coeffs = [1] * 12
                candidates.append(list(zip(exps, coeffs)))
                coeffs_alt = [1 if i % 2 == 0 else -1 for i in range(12)]
                candidates.append(list(zip(exps, coeffs_alt)))
    
    # Random starts
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        exps = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.choice([-1, 1]) for _ in range(12)]
        candidates.append(list(zip(exps, coeffs)))
    
    # A+B decomposition candidates
    for _ in range(30):
        if time.time() > deadline - 10:
            break
        all_exps = sorted(random.sample(range(0, 41), 12))
        a_exps = all_exps[:6]
        b_exps = all_exps[6:]
        for sign_pattern in range(4):
            a_coeffs = []
            b_coeffs = []
            for i in range(6):
                if sign_pattern == 0:
                    a_coeffs.append(1)
                    b_coeffs.append(1)
                elif sign_pattern == 1:
                    a_coeffs.append(1 if i % 2 == 0 else -1)
                    b_coeffs.append(1 if i % 2 == 0 else -1)
                elif sign_pattern == 2:
                    a_coeffs.append(1)
                    b_coeffs.append(-1)
                else:
                    a_coeffs.append(1 if i < 3 else -1)
                    b_coeffs.append(1 if i < 3 else -1)
            candidates.append(list(zip(a_exps, a_coeffs)) + list(zip(b_exps, b_coeffs)))

    # Deduplicate
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    # Run local search on each candidate
    for c in unique_candidates:
        if time.time() > deadline:
            break
        local_search_on_pairs(c)

    # --- Phase 3: Additional product-of-three with varied splits ---
    for restart in range(30):
        if time.time() > deadline - 5:
            break
        all_exps = sorted(random.sample(range(0, 41), 12))
        exps1 = all_exps[0:4]
        exps2 = all_exps[4:8]
        exps3 = all_exps[8:12]
        _, _ = optimize_product_structure_v2(exps1, exps2, exps3)

    # If no valid candidate found, return a fallback
    if best_candidate is None:
        best_candidate = [1] * 12

    return best_candidate
