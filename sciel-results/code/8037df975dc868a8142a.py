import time
import itertools

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit

    best_poly = None
    best_score = float('inf')

    def count_square_terms(d):
        """Count nonzero terms in the exact integer square of a sparse polynomial."""
        if not d:
            return 0
        square = {}
        items = list(d.items())
        n = len(items)
        for i in range(n):
            ei, ci = items[i]
            for j in range(i, n):
                ej, cj = items[j]
                key = ei + ej
                if i == j:
                    square[key] = square.get(key, 0) + ci * ci
                else:
                    square[key] = square.get(key, 0) + 2 * ci * cj
        return sum(1 for v in square.values() if v != 0)

    def validate_and_score(d):
        """Validate constraints and return score if valid, else None."""
        if not d:
            return None
        max_exp = max(d.keys())
        if max_exp > 40:
            return None
        n = max_exp + 1
        coeffs_list = [0] * n
        for e, c in d.items():
            coeffs_list[e] = c
        # Trim trailing zeros
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
        return count_square_terms(d)

    def dict_to_list(d):
        if not d:
            return [1]
        max_exp = max(d.keys())
        lst = [0] * (max_exp + 1)
        for e, c in d.items():
            lst[e] = c
        return lst

    # ====================================================================
    # STRATEGY: Exhaustive search over all 2^12 sign patterns for consecutive exponents.
    # This is fast (4096 iterations) and guarantees the best result in the 
    # +/-1 coefficient space for the 0..11 exponent structure.
    # ====================================================================
    exps_consec = list(range(12))
    for signs in itertools.product([-1, 1], repeat=12):
        if time.time() > deadline:
            break
        d = {exps_consec[i]: signs[i] for i in range(12)}
        score = validate_and_score(d)
        if score is not None and score < best_score:
            best_score = score
            best_poly = dict_to_list(d)

    # ====================================================================
    # STRATEGY 2: Search over small coefficient sets {-2, -1, 1, 2} with consecutive exponents.
    # This expands the search space to allow for more complex cancellations.
    # We limit the search to a manageable subset by fixing the first few coefficients
    # or using a heuristic to prune.
    # ====================================================================
    # To keep it fast, we'll search a subset:
    # Fix the first 6 coefficients to +1, and search the last 6 in {-1, 1}.
    # This is 2^6 = 64 patterns, very fast.
    for signs in itertools.product([-1, 1], repeat=6):
        if time.time() > deadline:
            break
        coeffs = [1] * 6 + list(signs)
        d = {i: coeffs[i] for i in range(12)}
        score = validate_and_score(d)
        if score is not None and score < best_score:
            best_score = score
            best_poly = dict_to_list(d)

    # Search with first 4 fixed to +1, next 4 in {-1, 1}, last 4 in {-1, 1}
    for signs in itertools.product([-1, 1], repeat=8):
        if time.time() > deadline:
            break
        coeffs = [1] * 4 + list(signs)
        d = {i: coeffs[i] for i in range(12)}
        score = validate_and_score(d)
        if score is not None and score < best_score:
            best_score = score
            best_poly = dict_to_list(d)

    # ====================================================================
    # STRATEGY 3: Try some known good patterns from literature or manual inspection.
    # For example, the alternating pattern (1, -1, 1, -1, ...) is known to be good.
    # Let's explicitly check it.
    # ====================================================================
    alt_coeffs = [1 if i % 2 == 0 else -1 for i in range(12)]
    d = {i: alt_coeffs[i] for i in range(12)}
    score = validate_and_score(d)
    if score is not None and score < best_score:
        best_score = score
        best_poly = dict_to_list(d)

    # ====================================================================
    # STRATEGY 4: Final Fallback & Validation
    # ====================================================================
    if best_poly is None:
        # Fallback: 12 ones
        best_poly = [1] * 12
    
    # Ensure return format is correct
    final_d = {i: c for i, c in enumerate(best_poly) if c != 0}
    if len(final_d) != 12:
        best_poly = [1] * 12

    return best_poly
