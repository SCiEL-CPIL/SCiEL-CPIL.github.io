import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    best_result = [1] * 12  # fallback
    best_nonzero = 144  # worst case
    
    def count_nonzero_square(coeffs, exponents):
        """Count nonzero terms in the square of sum(c_i * x^e_i) using numpy"""
        n = len(coeffs)
        # Build the full polynomial array
        max_deg = max(exponents)
        poly = np.zeros(2 * max_deg + 1, dtype=np.int64)
        for c, e in zip(coeffs, exponents):
            poly[e] = c
        # Compute square via convolution
        square = np.convolve(poly, poly)
        return int(np.count_nonzero(square))
    
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
        max_deg2 = 2 * max_deg
        # The leading term of P^2 is (c_last)^2 where c_last is coeff at max_deg
        # This is always nonzero if c_last != 0, which we already checked
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
    
    # Expanded exponent patterns with more variety
    exponent_patterns = [
        list(range(12)),                          # 0,1,2,...,11
        list(range(0, 24, 2)),                    # 0,2,4,...,22
        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 20],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 1, 3, 5, 6, 8, 10, 11, 13, 15, 17, 19],
        # New patterns: clustered
        [0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17],
        [0, 1, 2, 3, 4, 14, 15, 16, 17, 18, 19, 20],
        [0, 2, 4, 6, 14, 16, 18, 20, 22, 24, 26, 28],
        [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21],
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33],
        [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17],
        [0, 2, 4, 7, 9, 11, 14, 16, 18, 21, 23, 25],
        [0, 1, 3, 4, 6, 8, 9, 11, 13, 14, 16, 18],
        [0, 1, 2, 3, 4, 5, 6, 17, 18, 19, 20, 21],
        [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17],
    ]
    
    # Expanded coefficient sets
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
        # New coefficient patterns
        [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12],
        [1, 2, 3, 4, -5, -6, -7, -8, 9, 10, 11, 12],
        [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6],
        [1, 3, -2, 4, -3, 5, -4, 6, -5, 7, -6, 8],
        [1, -3, 2, -4, 3, -5, 4, -6, 5, -7, 6, -8],
        [1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
        [1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
        [2, -2, 2, -2, 2, -2, 2, -2, 2, -2, 2, -2],
        [1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12],
        [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12],
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
    
    # Strategy 2: Randomized local search
    if time.time() < deadline:
        rng = random.Random(42)
        for exponents in exponent_patterns[:8]:
            if time.time() > deadline:
                break
            max_deg = max(exponents)
            for _ in range(200):
                if time.time() > deadline:
                    break
                # Generate random coefficients with small magnitudes
                coeffs = []
                for i in range(12):
                    mag = rng.choice([1, 1, 1, 2, 2, 3, 4, 5])
                    sign = 1 if rng.random() < 0.5 else -1
                    coeffs.append(sign * mag)
                
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
    
    # Strategy 3: Symmetric coefficient patterns on arithmetic progression
    if time.time() < deadline:
        for step in [1, 2, 3]:
            exponents = [i * step for i in range(12)]
            max_deg = 2 * exponents[-1]
            if max_deg + 1 > 41:
                continue
            
            # Try various symmetric patterns
            for pattern_type in range(6):
                for base in range(1, 4):
                    coeffs = []
                    for i in range(12):
                        if pattern_type == 0:
                            # Alternating
                            c = base * ((-1) ** i)
                        elif pattern_type == 1:
                            # Symmetric: c_i = c_{11-i}
                            c = base * ((-1) ** (i % 2))
                        elif pattern_type == 2:
                            # Anti-symmetric: c_i = -c_{11-i}
                            c = base * ((-1) ** (i // 2))
                        elif pattern_type == 3:
                            # Triangular
                            c = base * (1 + (i % 3)) * ((-1) ** i)
                        elif pattern_type == 4:
                            # Sawtooth
                            c = base * ((i % 4) - 1) * ((-1) ** i)
                        else:
                            # Mixed
                            c = base * ((i % 3) - 1) * ((-1) ** (i // 3))
                        coeffs.append(c)
                    
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
    
    # Strategy 4: Try P = A(x) + x^k * A(x) where A has 6 terms
    if time.time() < deadline:
        for k in range(1, 12):
            for _ in range(50):
                if time.time() > deadline:
                    break
                A = []
                for i in range(6):
                    mag = random.choice([1, 1, 1, 2, 2, 3])
                    sign = 1 if random.random() < 0.5 else -1
                    A.append(sign * mag)
                
                # P has terms at exponents 0..5 and k..k+5
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
    
    # Strategy 5: Local search refinement on best found
    if time.time() < deadline:
        # Extract exponents and coeffs from best_result
        nz_indices = [i for i, c in enumerate(best_result) if c != 0]
        if len(nz_indices) == 12:
            best_exponents = nz_indices
            best_coeffs = [best_result[i] for i in nz_indices]
            
            # Try perturbing coefficients
            for _ in range(500):
                if time.time() > deadline:
                    break
                coeffs = best_coeffs[:]
                # Perturb 1-3 random coefficients
                num_perturb = random.randint(1, 3)
                for _ in range(num_perturb):
                    idx = random.randint(0, 11)
                    old_c = coeffs[idx]
                    new_c = old_c + random.choice([-2, -1, 1, 2])
                    if new_c != 0 and -64 <= new_c <= 64:
                        coeffs[idx] = new_c
                
                if any(c == 0 for c in coeffs):
                    continue
                if not is_valid_result(coeffs, best_exponents):
                    continue
                nz = count_nonzero_square(coeffs, best_exponents)
                if nz < best_nonzero:
                    best_nonzero = nz
                    result = coeffs_from_state((coeffs, best_exponents))
                    if len(result) <= 41 and result[-1] != 0:
                        best_result = result
                        best_coeffs = coeffs[:]
    
    # Strategy 6: Try more random exponent patterns
    if time.time() < deadline:
        rng = random.Random(123)
        for _ in range(100):
            if time.time() > deadline:
                break
            # Generate random sorted exponents with max <= 20
            exponents = sorted(rng.sample(range(0, 21), 12))
            if max(exponents) * 2 + 1 > 41:
                continue
            
            # Try a few coefficient patterns
            for _ in range(5):
                if time.time() > deadline:
                    break
                coeffs = []
                for i in range(12):
                    mag = rng.choice([1, 1, 1, 2, 2, 3])
                    sign = 1 if rng.random() < 0.5 else -1
                    coeffs.append(sign * mag)
                
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
    
    # Ensure result is valid
    if len(best_result) > 41 or best_result[-1] == 0:
        best_result = [1] * 12
    nz_count = sum(1 for c in best_result if c != 0)
    if nz_count != 12:
        best_result = [1] * 12
    if any(c < -64 or c > 64 for c in best_result):
        best_result = [1] * 12
    
    return best_result
