import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    best_result = [1] * 12  # fallback
    best_nonzero = 144  # worst case
    
    def count_nonzero_square(coeffs, exponents):
        """Count nonzero terms in the square of sum(c_i * x^e_i)"""
        square = defaultdict(int)
        n = len(coeffs)
        for i in range(n):
            for j in range(i, n):
                deg = exponents[i] + exponents[j]
                if i == j:
                    square[deg] += coeffs[i] * coeffs[j]
                else:
                    square[deg] += 2 * coeffs[i] * coeffs[j]
        return sum(1 for v in square.values() if v != 0)
    
    def is_valid_result(coeffs, exponents):
        if len(coeffs) != 12:
            return False
        if any(c == 0 for c in coeffs):
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        if any(e < 0 for e in exponents):
            return False
        # Check length <= 41
        max_deg = max(exponents)
        if 2 * max_deg + 1 > 41:
            return False
        # Check nonzero final coefficient
        square = defaultdict(int)
        n = len(coeffs)
        for i in range(n):
            for j in range(i, n):
                deg = exponents[i] + exponents[j]
                if i == j:
                    square[deg] += coeffs[i] * coeffs[j]
                else:
                    square[deg] += 2 * coeffs[i] * coeffs[j]
        if square[max_deg * 2] == 0:
            return False
        return True
    
    def coeffs_from_state(state):
        """Convert state to coefficient list in ascending degree, length <= 41"""
        coeffs = [0] * (2 * max(state[1]) + 1)
        for c, e in zip(state[0], state[1]):
            coeffs[e] = c
        # Trim trailing zeros but keep nonzero final
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs
    
    # Strategy 1: Search over small coefficient sets with structured exponents
    # Try symmetric exponent patterns
    
    # Pattern: pairs that sum to constant
    # e.g., exponents: 0, 1, 2, ..., 11 (arithmetic)
    # or: 0, 2, 4, ..., 22
    # or: clustered patterns
    
    exponent_patterns = [
        list(range(12)),                          # 0,1,2,...,11
        list(range(0, 24, 2)),                    # 0,2,4,...,22
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 20],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 1, 3, 5, 6, 8, 10, 11, 13, 15, 17, 19],
    ]
    
    # Coefficient sets to try
    coeff_sets = [
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
        [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
        [1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
        [2, -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2],
        [1, 2, -1, -2, 1, 2, -1, -2, 1, 2, -1, -2],
        [3, -2, 1, -3, 3, -2, 1, -3, 3, -2, 1, -3],
        [1, 1, -2, 2, -1, -1, 2, -2, 1, 1, -2, 2],
        [2, 1, -1, -2, -1, 2, 1, -1, -2, -1, 2, 1],
    ]
    
    # First pass: quick evaluation of structured patterns
    for exponents in exponent_patterns:
        if time.time() > deadline:
            break
        for coeffs in coeff_sets:
            if not is_valid_result(coeffs, exponents):
                continue
            nz = count_nonzero_square(coeffs, exponents)
            if nz < best_nonzero:
                best_nonzero = nz
                result = coeffs_from_state((coeffs, exponents))
                if len(result) <= 41 and result[-1] != 0:
                    best_result = result
    
    # Strategy 2: Local search - start from a good pattern and perturb
    if time.time() < deadline:
        # Try to improve by searching nearby coefficient assignments
        for exponents in exponent_patterns[:5]:
            if time.time() > deadline:
                break
            # Try all sign patterns with small magnitudes
            for c0 in range(-4, 5):
                if c0 == 0:
                    continue
                for c1 in range(-4, 5):
                    if c1 == 0:
                        continue
                    for c2 in range(-4, 5):
                        if c2 == 0:
                            continue
                        for c3 in range(-4, 5):
                            if c3 == 0:
                                continue
                            if time.time() > deadline:
                                break
                            # Complete with a pattern
                            for pattern in range(4):
                                coeffs = [c0, c1, c2, c3]
                                for k in range(4, 12):
                                    if pattern == 0:
                                        coeffs.append((-1)**k * max(1, abs(coeffs[k % 4])))
                                    elif pattern == 1:
                                        coeffs.append(coeffs[k % 4])
                                    elif pattern == 2:
                                        coeffs.append(-coeffs[k % 4])
                                    else:
                                        coeffs.append(((-1)**(k // 4)) * coeffs[k % 4])
                                # Ensure no zeros
                                if any(c == 0 for c in coeffs):
                                    continue
                                if any(abs(c) > 64 for c in coeffs):
                                    continue
                                if not is_valid_result(coeffs, exponents):
                                    continue
                                nz = count_nonzero_square(coeffs, exponents)
                                if nz < best_nonzero:
                                    best_nonzero = nz
                                    result = coeffs_from_state((coeffs, exponents))
                                    if len(result) <= 41 and result[-1] != 0:
                                        best_result = result
    
    # Strategy 3: Try to find patterns with more cancellation
    # Use the fact that if exponents are in arithmetic progression,
    # e_i + e_j = i + j (times step), so many collisions
    
    if time.time() < deadline:
        # Focus on arithmetic progression exponents
        for step in [1, 2, 3]:
            exponents = [i * step for i in range(12)]
            max_deg = 2 * exponents[-1]
            if max_deg + 1 > 41:
                continue
            
            # For arithmetic progression, sum s = e_i + e_j = (i+j)*step
            # For each s, pairs with i+j = s/step (if s divisible by step)
            # We want sum of 2*c_i*c_j over valid pairs to be 0
            
            # Try symmetric coefficients: c_i = c_{11-i} or c_i = -c_{11-i}
            for sign_pattern in range(128):  # 7-bit pattern for first 7, mirror
                coeffs = []
                for i in range(12):
                    bit = (sign_pattern >> (i % 7)) & 1
                    mag = 1 + (i % 3)  # small magnitudes
                    if bit:
                        coeffs.append(mag)
                    else:
                        coeffs.append(-mag)
                
                if any(c == 0 for c in coeffs):
                    continue
                if not is_valid_result(coeffs, exponents):
                    continue
                nz = count_nonzero_square(coeffs, exponents)
                if nz < best_nonzero:
                    best_nonzero = nz
                    result = coeffs_from_state((coeffs, exponents))
                    if len(result) <= 41 and result[-1] != 0:
                        best_result = result
    
    # Strategy 4: Try specific known-good structures
    # If P = (1-x)^6 + x^6*(1-x)^6, that's 12 terms? No, (1-x)^6 has 7 terms.
    # Try P = A(x) + x^k * A(x) where A has 6 terms
    if time.time() < deadline:
        for k in range(1, 10):
            for a0 in range(-3, 4):
                if a0 == 0:
                    continue
                for a1 in range(-3, 4):
                    if a1 == 0:
                        continue
                    for a2 in range(-3, 4):
                        if a2 == 0:
                            continue
                        for a3 in range(-3, 4):
                            if a3 == 0:
                                continue
                                for a4 in range(-3, 4):
                                    if a4 == 0:
                                        continue
                                    for a5 in range(-3, 4):
                                        if a5 == 0:
                                            continue
                                        A = [a0, a1, a2, a3, a4, a5]
                                        # P = A(x) + x^k * A(x)
                                        coeffs = A + [0] * (k - 1) + A if k > 6 else A[:k] + A[k:]
                                        # Actually, P has terms at exponents 0..5 and k..k+5
                                        # If k <= 5, there's overlap; need to handle
                                        if k <= 5:
                                            # Merge
                                            full_coeffs = [0] * (max(6, k + 6))
                                            for i in range(6):
                                                full_coeffs[i] += A[i]
                                                full_coeffs[k + i] += A[i]
                                            # Remove zeros
                                            terms = [(i, c) for i, c in enumerate(full_coeffs) if c != 0]
                                            if len(terms) != 12:
                                                continue
                                            coeffs = [c for _, c in terms]
                                            exponents = [e for e, _ in terms]
                                        else:
                                            coeffs = A + A
                                            exponents = list(range(6)) + list(range(k, k + 6))
                                        
                                        if any(abs(c) > 64 for c in coeffs):
                                            continue
                                        if not is_valid_result(coeffs, exponents):
                                            continue
                                        nz = count_nonzero_square(coeffs, exponents)
                                        if nz < best_nonzero:
                                            best_nonzero = nz
                                            result = coeffs_from_state((coeffs, exponents))
                                            if len(result) <= 41 and result[-1] != 0:
                                                best_result = result
                                        if time.time() > deadline:
                                            break
                                    if time.time() > deadline:
                                        break
                                if time.time() > deadline:
                                    break
                            if time.time() > deadline:
                                break
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
    
    # Ensure result is valid
    if not is_valid_result(best_result, None) if False else True:
        # Validate best_result as a coefficient list
        if len(best_result) > 41 or best_result[-1] == 0:
            best_result = [1] * 12
        # Check exactly 12 nonzero terms
        nz_count = sum(1 for c in best_result if c != 0)
        if nz_count != 12:
            best_result = [1] * 12
        # Check coefficient bounds
        if any(c < -64 or c > 64 for c in best_result):
            best_result = [1] * 12
    
    return best_result
