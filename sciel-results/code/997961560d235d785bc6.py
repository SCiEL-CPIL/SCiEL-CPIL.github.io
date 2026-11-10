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
        max_deg = max(exponents)
        poly = np.zeros(2 * max_deg + 1, dtype=np.int64)
        for c, e in zip(coeffs, exponents):
            poly[e] = c
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
    
    # Expanded exponent patterns
    exponent_patterns = [
        list(range(12)),
        list(range(0, 24, 2)),
        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 20],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 1, 3, 5, 6, 8, 10, 11, 13, 15, 17, 19],
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
    
    # Coefficient sets
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
        [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12],
        [1, 2, 3, 4, -5, -6, -7, -8, 9, 10, 11, 12],
        [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6],
        [1, 3, -2, 4, -3, 5, -4, 6, -5, 7, -6, 8],
        [1, -3, 2, -4, 3, -5, 4, -6, 5, -7, 6, -8],
        [1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
        [1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
        [2, -2, 2, -2, 2, -2, 2, -2, 2, -2, 2, -2],
        [1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12],
    ]
    
    # First pass: quick evaluation of structured patterns
    for exponents in exponent_patterns:
        if time.time() > deadline:
            break
        for coeffs in coeff_sets:
            if not is_valid_result(coeffs, exponents):
                continue
            nz = count_nonzero_square(coeffs, exponents)
            update_best(coeffs, exponents, nz)
    
    # GRAPH-BASED STRATEGY: Build collision graph and use branch-and-bound
    # over sign assignments to enforce cancellations
    if time.time() < deadline:
        for exponents in exponent_patterns:
            if time.time() > deadline:
                break
            
            # Build collision structure: for each degree d in P^2, which pairs (i,j) contribute
            n = 12
            max_deg = max(exponents)
            degree_pairs = defaultdict(list)
            for i in range(n):
                for j in range(i, n):
                    d = exponents[i] + exponents[j]
                    degree_pairs[d].append((i, j))
            
            # Magnitude patterns to try
            mag_patterns = [
                [1]*12,
                [1,1,1,1,1,1,2,2,2,2,2,2],
                [1,1,1,1,1,2,2,2,2,2,2,2],
                [1,1,1,2,2,2,2,2,2,2,2,2],
                [1,1,2,2,2,2,2,2,2,2,2,2],
                [1,2,2,2,2,2,2,2,2,2,2,2],
                [1,1,1,1,2,2,2,2,2,2,2,2],
                [1,1,2,1,2,1,2,1,2,1,2,1],
                [1,2,1,2,1,2,1,2,1,2,1,2],
            ]
            
            for mags in mag_patterns:
                if time.time() > deadline:
                    break
                
                # Pre-compute constraint structure
                # For each degree, store list of (i, j, weight) where weight = w*m_i*m_j
                constraints = []
                for d in sorted(degree_pairs.keys()):
                    pairs = degree_pairs[d]
                    if len(pairs) <= 1:
                        continue
                    constraint = []
                    for (i, j) in pairs:
                        w = 1 if i == j else 2
                        weight = w * mags[i] * mags[j]
                        constraint.append((i, j, weight))
                    constraints.append(constraint)
                
                if not constraints:
                    continue
                
                # Local search over sign assignments
                rng = random.Random(hash(tuple(mags)) % 10000)
                for trial in range(30):
                    if time.time() > deadline:
                        break
                    
                    # Random initial sign assignment
                    signs = [1 if rng.random() < 0.5 else -1 for _ in range(12)]
                    
                    def eval_signs(s):
                        """Evaluate number of nonzero terms in P^2 for given signs"""
                        total_nz = 0
                        for constraint in constraints:
                            val = 0
                            for (i, j, weight) in constraint:
                                val += weight * s[i] * s[j]
                            if val != 0:
                                total_nz += 1
                        return total_nz
                    
                    # Also count diagonal-only degrees (always nonzero)
                    diag_only_nz = 0
                    for d in degree_pairs:
                        pairs = degree_pairs[d]
                        if len(pairs) == 1:
                            diag_only_nz += 1
                    
                    current_nz = eval_signs(signs) + diag_only_nz
                    
                    # Local search: flip signs to reduce nonzero count
                    improved = True
                    while improved and time.time() < deadline:
                        improved = False
                        for idx in range(12):
                            signs[idx] = -signs[idx]
                            new_nz = eval_signs(signs) + diag_only_nz
                            if new_nz < current_nz:
                                current_nz = new_nz
                                improved = True
                            else:
                                signs[idx] = -signs[idx]
                    
                    if current_nz < best_nonzero:
                        coeffs = [mags[i] * signs[i] for i in range(12)]
                        if all(c != 0 for c in coeffs) and all(-64 <= c <= 64 for c in coeffs):
                            update_best(coeffs, exponents, current_nz)
    
    # Strategy 2: Randomized local search on coefficients
    if time.time() < deadline:
        rng = random.Random(42)
        for exponents in exponent_patterns[:10]:
            if time.time() > deadline:
                break
            for _ in range(150):
                if time.time() > deadline:
                    break
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
                update_best(coeffs, exponents, nz)
    
    # Strategy 3: Local search refinement on best found
    if time.time() < deadline:
        nz_indices = [i for i, c in enumerate(best_result) if c != 0]
        if len(nz_indices) == 12:
            best_exponents = nz_indices
            best_coeffs = [best_result[i] for i in nz_indices]
            
            for _ in range(800):
                if time.time() > deadline:
                    break
                coeffs = best_coeffs[:]
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
    
    # Strategy 4: Random exponent patterns with targeted search
    if time.time() < deadline:
        rng = random.Random(123)
        for _ in range(80):
            if time.time() > deadline:
                break
            exponents = sorted(rng.sample(range(0, 21), 12))
            if max(exponents) * 2 + 1 > 41:
                continue
            
            for _ in range(8):
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
                update_best(coeffs, exponents, nz)
    
    # Strategy 5: Try P = A(x) + x^k * A(x) where A has 6 terms
    if time.time() < deadline:
        for k in range(1, 12):
            for _ in range(40):
                if time.time() > deadline:
                    break
                A = []
                for i in range(6):
                    mag = random.choice([1, 1, 1, 2, 2, 3])
                    sign = 1 if random.random() < 0.5 else -1
                    A.append(sign * mag)
                
                if k <= 5:
                    full_coeffs = [0] * (max(6, k + 6))
                    for i in range(6):
                        full_coeffs[i] += A[i]
                        full_coeffs[k + i] += A[i]
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
                update_best(coeffs, exponents, nz)
    
    # Ensure result is valid
    if len(best_result) > 41 or best_result[-1] == 0:
        best_result = [1] * 12
    nz_count = sum(1 for c in best_result if c != 0)
    if nz_count != 12:
        best_result = [1] * 12
    if any(c < -64 or c > 64 for c in best_result):
        best_result = [1] * 12
    
    return best_result
