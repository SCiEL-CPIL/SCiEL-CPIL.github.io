import time
import itertools
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    best_count = float('inf')
    best_coeffs = [1]*12

    def count_nonzero(square):
        return int(np.count_nonzero(square))

    def square_coeffs(coeffs_list):
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
        max_deg = max(exponents)
        result = [0] * (max_deg + 1)
        for exp, val in zip(exponents, values):
            result[exp] = val
        return result

    def try_polynomial(exponents, values):
        nonlocal best_count, best_coeffs
        if len(exponents) != 12:
            return
        if max(exponents) >= 41:
            return
        if any(v == 0 for v in values) or any(abs(v) > 64 for v in values):
            return
        coeffs_list = list(zip(exponents, values))
        square = square_coeffs(coeffs_list)
        count = count_nonzero(square)
        if count < best_count:
            best_count = count
            best_coeffs = coeffs_from_exponents(exponents, values)

    # Phase 1: Brute force consecutive exponents 0..11 with ±1
    for bits in range(1 << 12):
        if time.time() > deadline:
            break
        coeffs = [1 if (bits >> i) & 1 else -1 for i in range(12)]
        square = np.convolve(coeffs, coeffs)
        count = count_nonzero(square)
        if count < best_count:
            best_count = count
            best_coeffs = coeffs

    # Phase 2: Inverted search - fix sparse Q support, solve for P
    # Strategy: Q = P^2 has k nonzero terms. Each Q_j = sum_{i<=j} c_i * c_{j-i} (with factors)
    # For sparse Q, we can try to find P that produces it.
    # 
    # Key insight: If Q has support {d_1, d_2, ..., d_k} with values q_1,...,q_k,
    # and P has support {e_1,...,e_12}, then Q's support is {e_i + e_j : i<=j}.
    # 
    # For 12 terms in P, P^2 has at most 78 distinct degree sums (12*13/2).
    # To minimize nonzero terms in P^2, we want many collisions: e_i + e_j = e_k + e_l for different pairs.
    #
    # Arithmetic progressions maximize collisions. For P with exponents {a, a+d, a+2d, ..., a+11d},
    # P^2 has exponents {2a+kd : k=0..22}, so 23 distinct degrees.
    # With signs chosen to cancel, we can reduce further.
    
    # Phase 2a: Structured AP search with optimized sign patterns
    for gap in [1, 2, 3, 4, 5, 6, 7, 8]:
        if time.time() > deadline:
            break
        for base in [0, 1, 2]:
            if time.time() > deadline:
                break
            exponents = [base + i * gap for i in range(12)]
            if max(exponents) >= 41:
                continue
            
            # Try all sign patterns for this AP (2^12 = 4096)
            for bits in range(1 << 12):
                if time.time() > deadline:
                    break
                signs = [1 if (bits >> i) & 1 else -1 for i in range(12)]
                try_polynomial(exponents, signs)
    
    # Phase 2b: Try coefficient values beyond ±1 for APs
    for gap in [2, 3, 4]:
        if time.time() > deadline:
            break
        for base in [0, 1]:
            if time.time() > deadline:
                break
            exponents = [base + i * gap for i in range(12)]
            if max(exponents) >= 41:
                continue
            
            # Try small coefficient patterns
            for val_pattern in [
                [1,1,1,1,1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1,1,1,1,-1],
                [1,1,1,1,1,1,1,1,1,1,-1,1],
                [1,1,1,1,1,1,1,1,1,-1,1,1],
                [1,1,1,1,1,1,1,1,-1,1,1,1],
                [1,1,1,1,1,1,1,-1,1,1,1,1],
                [1,1,1,1,1,1,-1,1,1,1,1,1],
                [1,1,1,1,1,-1,1,1,1,1,1,1],
                [1,1,1,1,-1,1,1,1,1,1,1,1],
                [1,1,1,-1,1,1,1,1,1,1,1,1],
                [1,1,-1,1,1,1,1,1,1,1,1,1],
                [1,-1,1,1,1,1,1,1,1,1,1,1],
                [-1,1,1,1,1,1,1,1,1,1,1,1],
            ]:
                if time.time() > deadline:
                    break
                try_polynomial(exponents, val_pattern)
    
    # Phase 2c: Inverted approach - try to construct P from desired Q
    # If Q = P^2 has very few terms, say 13 terms (minimum possible for 12-term P 
    # is 12 if P is a monomial times a 12-term polynomial with all cross terms canceling,
    # but that's impossible for 12 terms since P^2 has at least 12 diagonal terms)
    #
    # Actually, minimum nonzero terms in P^2 for 12-term P:
    # - Diagonal: 12 terms (c_i^2 at degree 2*e_i)
    # - Off-diagonal: c_i*c_j at degree e_i+e_j for i≠j
    # - Cancellation happens when e_i+e_j = e_k+e_l and c_i*c_j = -c_k*c_l
    #
    # For AP with gap d: degrees in P^2 are 2*base + k*d for k=0..22
    # Number of distinct degrees = 23
    # Each degree 2*base + k*d gets contributions from pairs (i,j) with i+j=k
    # For k < 12: pairs are (0,k), (1,k-1), ..., (k,0) -> k+1 pairs
    # For k >= 12: pairs are (k-11,11), (k-10,10), ..., (11,k-11) -> 23-k pairs
    #
    # With signs ±1, each pair contributes ±2 (off-diag) or ±1 (diag)
    # We want most degree coefficients to be 0.
    
    # Phase 2d: More systematic search with wider coefficient range
    import random
    rng = random.Random(42)
    
    # Random search with structured exponents
    for iteration in range(500000):
        if time.time() > deadline:
            break
        
        # Choose exponent structure
        struct_type = rng.random()
        
        if struct_type < 0.3:
            # Arithmetic progression
            base = rng.choice([0, 1, 2, 3])
            gap = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            exponents = [base + i * gap for i in range(12)]
        elif struct_type < 0.5:
            # Two interleaved APs
            base1 = rng.choice([0, 1, 2])
            base2 = base1 + rng.choice([1, 2, 3])
            gap = rng.choice([2, 3, 4, 5, 6])
            exponents = []
            for i in range(6):
                exponents.append(base1 + i * gap)
                exponents.append(base2 + i * gap)
            exponents.sort()
        elif struct_type < 0.7:
            # Symmetric around center
            center = rng.randint(10, 30)
            offsets = []
            for i in range(6):
                off = rng.randint(1, 15)
                offsets.append(off)
                offsets.append(-off)
            rng.shuffle(offsets)
            exponents = sorted([center + o for o in offsets])
            # Ensure non-negative and distinct
            if min(exponents) < 0 or len(set(exponents)) != 12:
                continue
        elif struct_type < 0.85:
            # Random with clustering
            base = rng.randint(0, 5)
            exponents = [base]
            for i in range(11):
                gap = rng.choice([1, 1, 2, 2, 3, 4, 5])
                exponents.append(exponents[-1] + gap)
        else:
            # Fully random
            exponents = sorted(rng.sample(range(0, 41), 12))
        
        if max(exponents) >= 41 or len(set(exponents)) != 12:
            continue
        
        # Choose coefficients
        coeff_type = rng.random()
        if coeff_type < 0.4:
            values = [rng.choice([-1, 1]) for _ in range(12)]
        elif coeff_type < 0.6:
            values = [rng.choice([-2, -1, 1, 2]) for _ in range(12)]
        elif coeff_type < 0.8:
            values = [rng.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
        else:
            values = [rng.randint(-5, 5) for _ in range(12)]
            values = [v if v != 0 else 1 for v in values]
        
        try_polynomial(exponents, values)
    
    # Phase 3: Local search - perturb best solution
    if best_count < float('inf') and time.time() < deadline:
        # Convert best_coeffs back to (exponents, values)
        best_exponents = [i for i, c in enumerate(best_coeffs) if c != 0]
        best_values = [c for c in best_coeffs if c != 0]
        
        for iteration in range(200000):
            if time.time() > deadline:
                break
            
            new_exponents = list(best_exponents)
            new_values = list(best_values)
            
            # Apply 1-2 perturbations
            n_perturb = rng.randint(1, 2)
            for _ in range(n_perturb):
                pert_type = rng.random()
                if pert_type < 0.3:
                    # Change a coefficient value
                    idx = rng.randint(0, 11)
                    new_values[idx] = rng.choice([-3, -2, -1, 1, 2, 3])
                elif pert_type < 0.5:
                    # Shift an exponent
                    idx = rng.randint(0, 11)
                    delta = rng.choice([-2, -1, 1, 2])
                    new_exponents[idx] += delta
                elif pert_type < 0.7:
                    # Swap two exponents
                    i, j = rng.sample(range(12), 2)
                    new_exponents[i], new_exponents[j] = new_exponents[j], new_exponents[i]
                elif pert_type < 0.85:
                    # Change sign
                    idx = rng.randint(0, 11)
                    new_values[idx] = -new_values[idx]
                else:
                    # Replace one exponent with random
                    idx = rng.randint(0, 11)
                    new_exponents[idx] = rng.randint(0, 40)
            
            # Validate
            if len(set(new_exponents)) != 12:
                continue
            if min(new_exponents) < 0 or max(new_exponents) >= 41:
                continue
            if any(abs(v) > 64 or v == 0 for v in new_values):
                continue
            
            try_polynomial(sorted(new_exponents), [v for _, v in sorted(zip(new_exponents, new_values))])
    
    # Ensure valid output
    if len(best_coeffs) > 41:
        best_coeffs = best_coeffs[:41]
    if best_coeffs[-1] == 0:
        for i in range(len(best_coeffs) - 1, -1, -1):
            if best_coeffs[i] != 0:
                best_coeffs = best_coeffs[:i+1]
                break
    if len(best_coeffs) == 0:
        best_coeffs = [1]

    return best_coeffs
