import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s safety margin for 90s limit
    
    best_count = float('inf')
    best_coeffs = [1]*12
    
    def count_nonzero(square):
        return int(np.count_nonzero(square))
    
    def square_coeffs(coeffs_list):
        """Compute exact integer square of polynomial given as list of (exp, coeff) pairs."""
        n_terms = len(coeffs_list)
        max_deg = max(e for e, c in coeffs_list)
        square_len = 2 * max_deg + 1
        square = np.zeros(square_len, dtype=np.int64)
        for i in range(n_terms):
            ei, ci = coeffs_list[i]
            for j in range(i, n_terms):
                ej, cj = coeffs_list[j]
                prod = ci * cj
                if i == j:
                    square[ei + ej] += prod
                else:
                    square[ei + ej] += 2 * prod
        return square
    
    def coeffs_from_exponents(exponents, values):
        """Build full coefficient list from exponents and values."""
        max_deg = max(exponents)
        result = [0] * (max_deg + 1)
        for exp, val in zip(exponents, values):
            result[exp] = val
        return result
    
    # Phase 1: Search consecutive exponents 0..11 with ±1 (fast baseline)
    for bits in range(1 << 12):
        if time.time() > deadline:
            break
        coeffs = [1 if (bits >> i) & 1 else -1 for i in range(12)]
        square = np.convolve(coeffs, coeffs)
        count = count_nonzero(square)
        if count < best_count:
            best_count = count
            best_coeffs = coeffs
    
    # Phase 2: Try some structured patterns with gaps
    # Pattern: alternating signs with specific gaps
    gap_patterns = [
        # (exponents, sign_pattern)
        (list(range(0, 24, 2)), [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]),
        (list(range(0, 24, 2)), [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1]),
        (list(range(0, 24, 2)), [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1]),
        (list(range(0, 36, 3)), [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]),
        (list(range(0, 36, 3)), [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1]),
    ]
    
    for exponents, signs in gap_patterns:
        if time.time() > deadline:
            break
        coeffs_list = list(zip(exponents, signs))
        square = square_coeffs(coeffs_list)
        count = count_nonzero(square)
        if count < best_count:
            best_count = count
            best_coeffs = coeffs_from_exponents(exponents, signs)
    
    # Phase 3: Random search with gaps and small coefficients
    import random
    rng = random.Random(42)
    
    for iteration in range(200000):
        if time.time() > deadline:
            break
        
        # Choose random exponents with some structure
        base = rng.choice([0, 1, 2])
        gap = rng.choice([1, 2, 3, 4])
        exponents = [base + i * gap for i in range(12)]
        
        # Ensure length <= 41
        if max(exponents) >= 41:
            continue
        
        # Choose small integer coefficients
        if rng.random() < 0.5:
            values = [rng.choice([-1, 1]) for _ in range(12)]
        else:
            values = [rng.choice([-2, -1, 1, 2]) for _ in range(12)]
        
        # Ensure all coefficients in [-64, 64] and exactly 12 nonzero
        if any(v == 0 for v in values) or any(abs(v) > 64 for v in values):
            continue
        
        coeffs_list = list(zip(exponents, values))
        square = square_coeffs(coeffs_list)
        count = count_nonzero(square)
        
        if count < best_count:
            best_count = count
            best_coeffs = coeffs_from_exponents(exponents, values)
    
    # Ensure valid output
    if len(best_coeffs) > 41:
        best_coeffs = best_coeffs[:41]
    if best_coeffs[-1] == 0:
        # Find last nonzero
        for i in range(len(best_coeffs) - 1, -1, -1):
            if best_coeffs[i] != 0:
                best_coeffs = best_coeffs[:i+1]
                break
    if len(best_coeffs) == 0:
        best_coeffs = [1]
    
    return best_coeffs
