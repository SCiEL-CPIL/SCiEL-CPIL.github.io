import numpy as np
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    def count_nonzero(poly):
        return int(np.count_nonzero(poly))

    def poly_square(coeffs):
        return np.convolve(coeffs, coeffs)

    def evaluate(coeffs):
        sq = poly_square(coeffs)
        return count_nonzero(sq)

    def is_valid(coeffs):
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0:
            return False
        if coeffs[-1] == 0:
            return False
        nonzero_count = sum(1 for c in coeffs if c != 0)
        if nonzero_count != 12:
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        return True

    best_result = [1] * 12
    best_score = evaluate(best_result)  # 23 nonzero terms in square

    # Try some structured patterns
    # Pattern 1: powers of 2 spacing (geometric progression of exponents)
    # x^0 + x^2 + x^4 + ... + x^22 -> 12 terms, square has 23 terms (no cancellation)
    # We need cancellation. Let's try to find polynomials where square has fewer terms.

    # Key insight: if P(x) = sum a_i x^i, then P(x)^2 has terms at positions 2i and i+j.
    # Cancellation happens when a_i*a_j + a_k*a_l = 0 for same degree.

    # Let's try small coefficient values and random degree patterns
    # Strategy: pick 12 positions out of 0..40, assign coefficients from {-1, 1} or {-1, 0, 1}

    random.seed(42)

    def try_random():
        nonlocal best_result, best_score
        # Choose 12 distinct degree positions from 0 to 40
        degrees = sorted(random.sample(range(41), 12))
        # Ensure last position has nonzero coefficient (it does since we pick from range)
        # Assign coefficients
        coeff_values = random.choice([
            [-1, 1],
            [-1, 0, 1],
            [-2, -1, 0, 1, 2],
        ])
        coeffs = [0] * 41
        for d in degrees:
            coeffs[d] = random.choice(coeff_values)
        # Make sure no coefficient is zero at chosen positions
        for d in degrees:
            if coeffs[d] == 0:
                coeffs[d] = random.choice([-1, 1])
        # Trim trailing zeros
        while len(coeffs) > 0 and coeffs[-1] == 0:
            coeffs.pop()
        if len(coeffs) == 0 or len(coeffs) > 41:
            return
        # Check exactly 12 nonzero
        nz = sum(1 for c in coeffs if c != 0)
        if nz != 12:
            return
        if any(c < -64 or c > 64 for c in coeffs):
            return
        score = evaluate(coeffs)
        if score < best_score:
            best_score = score
            best_result = coeffs[:]

    # Try structured approaches first
    # Approach: P(x) = x^a + x^b + ... where pairs cancel in square
    # Example: (1 + x + x^2)^2 = 1 + 2x + 3x^2 + 2x^3 + x^4 -> 5 terms
    # We want 12-term polynomial with sparse square.

    # Let's try: polynomial with symmetric structure
    # P(x) = sum_{i=0}^{11} c_i * x^{d_i}
    # If d_i + d_j = d_k + d_l and c_i*c_j = -c_k*c_l, then cancellation

    # Try specific known sparse-square constructions
    # (1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 + x^8 + x^9 + x^10 + x^11)^2 has 23 terms
    
    # Better: use signs to create cancellation
    # (1 - x + x^2 - x^3 + ... + x^11)^2
    # Let's compute this
    coeffs_alt = [1 if i % 2 == 0 else -1 for i in range(12)]
    if is_valid(coeffs_alt):
        s = evaluate(coeffs_alt)
        if s < best_score:
            best_score = s
            best_result = coeffs_alt[:]

    # Try (1 + x - x^2 + x^3 - x^4 + ...)^2
    for sign_pattern in range(8):
        coeffs_test = [0] * 41
        for i in range(12):
            # Alternate signs with some variation
            if (i // 2) % 2 == 0:
                coeffs_test[i] = 1
            else:
                coeffs_test[i] = -1
        # Trim
        while len(coeffs_test) > 0 and coeffs_test[-1] == 0:
            coeffs_test.pop()
        if is_valid(coeffs_test):
            s = evaluate(coeffs_test)
            if s < best_score:
                best_score = s
                best_result = coeffs_test[:]

    # Random search
    iterations = 0
    while time.time() < deadline:
        try_random()
        iterations += 1
        if iterations % 10000 == 0:
            if time.time() > deadline:
                break

    # Try local improvement: flip signs of individual terms
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current = best_result[:]
            for idx in range(len(current)):
                if current[idx] == 0:
                    continue
                # Try negating this coefficient
                old_val = current[idx]
                current[idx] = -old_val
                if is_valid(current):
                    s = evaluate(current)
                    if s < best_score:
                        best_score = s
                        best_result = current[:]
                        improved = True
                current[idx] = old_val

    return best_result
