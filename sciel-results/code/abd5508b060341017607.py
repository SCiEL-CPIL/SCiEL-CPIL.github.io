import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = 85.0

    best_poly = None
    best_score = float('inf')

    def count_square_terms(coeffs_dict):
        """Count nonzero terms in square given sparse dict {exp: coeff}."""
        if not coeffs_dict:
            return 0
        square = {}
        items = list(coeffs_dict.items())
        for i in range(len(items)):
            ei, ci = items[i]
            for j in range(i, len(items)):
                ej, cj = items[j]
                key = ei + ej
                if i == j:
                    val = ci * ci
                else:
                    val = 2 * ci * cj
                square[key] = square.get(key, 0) + val
        return sum(1 for v in square.values() if v != 0)

    def dict_to_list(d):
        """Convert sparse dict to dense list (ascending degree)."""
        if not d:
            return [0]
        max_exp = max(d.keys())
        lst = [0] * (max_exp + 1)
        for e, c in d.items():
            lst[e] = c
        return lst

    def evaluate_from_dict(coeffs_dict):
        """Validate and score a sparse polynomial representation."""
        if not coeffs_dict:
            return None
        max_exp = max(coeffs_dict.keys())
        if max_exp > 40:
            return None
        n = max_exp + 1
        coeffs_list = [0] * n
        for e, c in coeffs_dict.items():
            coeffs_list[e] = c
        while len(coeffs_list) > 1 and coeffs_list[-1] == 0:
            coeffs_list.pop()
        if len(coeffs_list) > 41:
            return None
        if coeffs_list[-1] == 0:
            return None
        nonzero_count = sum(1 for c in coeffs_list if c != 0)
        if nonzero_count != 12:
            return None
        for c in coeffs_list:
            if not (-64 <= c <= 64):
                return None
        score = count_square_terms(coeffs_dict)
        return score

    # ====================================================================
    # STRATEGY: Algebraic constructive approach
    #
    # Key insight: If exponents form an arithmetic progression with step k,
    # the square has support only at multiples of k (even degrees for k=2).
    # This reduces the problem to solving a system of quadratic equations
    # in the coefficients, where we want S_m = sum_{i+j=m} c_i c_j = 0
    # for as many m as possible.
    #
    # For 12 terms at exponents 0,2,4,...,22 (step 2), the square has
    # support at degrees 0,2,4,...,44 (23 positions). We want to zero out
    # as many of the 21 interior positions as possible.
    #
    # We'll do a targeted exhaustive search over small coefficient values
    # for this structured exponent set, plus some variants.
    # ====================================================================

    # --- Main strategy: Arithmetic progression with step 2 ---
    # Exponents: 0, 2, 4, ..., 22
    # Square support: 0, 2, ..., 44 (23 positions)
    # S_0 = c_0^2, S_44 = c_11^2 (always nonzero)
    # S_2, S_4, ..., S_42: 21 equations to try to zero out

    exps_ap2 = [2 * i for i in range(12)]  # 0,2,4,...,22

    # Try all combinations of coefficients from {-2,-1,1,2}
    # 4^12 = 16,777,216 -- too many for exhaustive, but we can be smart.

    # Smarter approach: fix c_0 = 1 (WLOG by scaling/sign), then search
    # over the remaining 11 coefficients from {-2,-1,1,2}
    # 4^11 = 4,194,304 -- still a lot, but each eval is fast.

    # Even smarter: use the structure. S_2 = 2*c_0*c_1 = 0 => c_1 = 0,
    # but we need all 12 nonzero. So S_2 can't be zeroed with c_0 != 0.
    # Actually S_2 = 2*c_0*c_1, so if c_0 and c_1 are both nonzero, S_2 != 0.
    # So we can't zero S_2. Similarly S_42 = 2*c_10*c_11 != 0.
    #
    # Let's just do a bounded random search with the AP structure,
    # but with much more iterations and a better coefficient range.

    random.seed(12345)

    # Phase 1: Random search over AP step-2 exponents with varied coefficients
    for iteration in range(200000):
        if time.time() > deadline:
            break

        d = {}
        for i in range(12):
            d[exps_ap2[i]] = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])

        score = evaluate_from_dict(d)
        if score is not None and score < best_score:
            best_score = score
            best_poly = dict_to_list(d)

    # Phase 2: Local search refinement on best found so far
    if best_poly is not None:
        d_best = {i: best_poly[i] for i in range(len(best_poly)) if best_poly[i] != 0}
        for restart in range(200):
            if time.time() > deadline:
                break
            d = dict(d_best)
            curr_score = best_score
            for step in range(3000):
                if time.time() > deadline:
                    break
                idx = random.randint(0, 11)
                e = exps_ap2[idx]
                old_val = d[e]
                new_val = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
                if new_val == old_val:
                    continue
                d[e] = new_val
                new_score = count_square_terms(d)
                if new_score < curr_score:
                    curr_score = new_score
                    if curr_score < best_score:
                        best_score = curr_score
                        best_poly = dict_to_list(d)
                        d_best = dict(d)
                else:
                    d[e] = old_val

    # Phase 3: Try AP step 1 (consecutive exponents 0..11)
    exps_ap1 = list(range(12))
    for iteration in range(100000):
        if time.time() > deadline:
            break
        d = {}
        for i in range(12):
            d[exps_ap1[i]] = random.choice([-3, -2, -1, 1, 2, 3])
        score = evaluate_from_dict(d)
        if score is not None and score < best_score:
            best_score = score
            best_poly = dict_to_list(d)

    # Phase 4: Try AP step 3 (exponents 0,3,6,...,33)
    exps_ap3 = [3 * i for i in range(12)]  # 0,3,...,33
    for iteration in range(100000):
        if time.time() > deadline:
            break
        d = {}
        for i in range(12):
            d[exps_ap3[i]] = random.choice([-3, -2, -1, 1, 2, 3])
        score = evaluate_from_dict(d)
        if score is not None and score < best_score:
            best_score = score
            best_poly = dict_to_list(d)

    # Phase 5: Try "two-gap" structure: 6 terms at even positions, 6 at odd
    # Exponents: 0,1,2,3,4,5, 20,21,22,23,24,25
    # This creates a large gap so A^2 and B^2 don't overlap,
    # but cross terms 2*A*B might have structure.
    for iteration in range(50000):
        if time.time() > deadline:
            break
        d = {}
        for i in range(6):
            d[i] = random.choice([-3, -2, -1, 1, 2, 3])
        for j in range(6):
            d[20 + j] = random.choice([-3, -2, -1, 1, 2, 3])
        if len(d) != 12:
            continue
        score = evaluate_from_dict(d)
        if score is not None and score < best_score:
            best_score = score
            best_poly = dict_to_list(d)

    # Phase 6: Structured sign patterns on AP step 2
    # c_i = (-1)^i * a_i where a_i are small integers
    for a_pattern in itertools.product([1, 2, 3], repeat=12):
        if time.time() > deadline:
            break
        d = {}
        for i in range(12):
            d[exps_ap2[i]] = ((-1) ** i) * a_pattern[i]
        score = evaluate_from_dict(d)
        if score is not None and score < best_score:
            best_score = score
            best_poly = dict_to_list(d)

    # Phase 7: Try specific algebraic constructions
    # P(x) = (1 + x^2 + x^4 + ... + x^22) - x*(1 + x^2 + ... + x^20)
    # = 1 - x + x^2 - x^3 + ... + x^20 - x^21 + x^22
    # That's 23 terms, too many.
    #
    # P(x) = (1 + x^2 + x^4 + x^6 + x^8 + x^10) - (x + x^3 + x^5 + x^7 + x^9 + x^11)
    # = 1 - x + x^2 - x^3 + x^4 - x^5 + x^6 - x^7 + x^8 - x^9 + x^10 - x^11
    # 12 terms, consecutive exponents. Already in phase 3 territory.

    # Try: P(x) = sum_{i=0}^{5} (-1)^i * x^{2i} + sum_{i=0}^{5} (-1)^{i+6} * x^{2i+13}
    # Two clusters with alternating signs
    d = {}
    for i in range(6):
        d[2 * i] = (-1) ** i
    for i in range(6):
        d[2 * i + 13] = (-1) ** (i + 6)
    if len(d) == 12:
        score = evaluate_from_dict(d)
        if score is not None and score < best_score:
            best_score = score
            best_poly = dict_to_list(d)

    # Final fallback
    if best_poly is None:
        best_poly = [1] * 12

    # Final validation
    final_d = {i: best_poly[i] for i in range(len(best_poly)) if best_poly[i] != 0}
    final_score = evaluate_from_dict(final_d)
    if final_score is None:
        best_poly = [1] * 12

    return best_poly
