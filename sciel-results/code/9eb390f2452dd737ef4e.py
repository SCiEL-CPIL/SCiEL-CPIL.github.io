"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    best_poly = None
    best_score = float('inf')
    
    # Helper: compute sparse square coefficient count
    def square_nonzero_count(coeffs):
        """Given list of (exponent, coefficient) pairs, return count of nonzero terms in square."""
        n = len(coeffs)
        if n == 0:
            return 0
        # Build dict of exponent -> coefficient for the square
        sq = {}
        for i in range(n):
            ei, ci = coeffs[i]
            for j in range(i, n):
                ej, cj = coeffs[j]
                exp = ei + ej
                if i == j:
                    sq[exp] = sq.get(exp, 0) + ci * ci
                else:
                    sq[exp] = sq.get(exp, 0) + 2 * ci * cj
        # Count nonzero
        return sum(1 for v in sq.values() if v != 0)
    
    # Helper: validate and trim
    def validate_and_trim(poly):
        """Ensure poly meets constraints, return trimmed list or None."""
        if not poly:
            return None
        # Trim trailing zeros
        while poly and poly[-1] == 0:
            poly = poly[:-1]
        if not poly:
            return None
        if len(poly) > 41:
            return None
        if poly[-1] == 0:
            return None
        nonzero_count = sum(1 for c in poly if c != 0)
        if nonzero_count != 12:
            return None
        if any(abs(c) > 64 for c in poly):
            return None
        return poly
    
    # Helper: build polynomial from sparse representation
    def build_poly(sparse_coeffs):
        """Build dense polynomial list from (exp, coeff) pairs."""
        if not sparse_coeffs:
            return None
        max_exp = max(e for e, c in sparse_coeffs)
        poly = [0] * (max_exp + 1)
        for exp, c in sparse_coeffs:
            poly[exp] = c
        return validate_and_trim(poly)
    
    # Helper: try a candidate and update best
    def try_candidate(sparse_coeffs):
        nonlocal best_poly, best_score
        score = square_nonzero_count(sparse_coeffs)
        if score < best_score:
            best_score = score
            poly = build_poly(sparse_coeffs)
            if poly is not None:
                best_poly = poly
        return score
    
    # Phase 1: AP with ±1 coefficients (fast exhaustive)
    for d in range(1, 4):
        max_a = 40 - 11 * d
        for a in range(0, max_a + 1):
            exponents = [a + i * d for i in range(12)]
            if exponents[-1] > 40:
                continue
            for signs_mask in range(1 << 12):
                if time.time() > deadline:
                    if best_poly is not None:
                        return best_poly
                    return [1]*12
                coeffs = [(exponents[i], 1 if (signs_mask >> i) & 1 else -1) for i in range(12)]
                try_candidate(coeffs)
    
    # Phase 2: Borwein-Erdelyi product construction
    # P(x) = A(x) * B(x) where A and B are sparse with small coeffs
    # This targets structures where cross-terms create cancellations
    random.seed(42)
    coeff_choices = [-2, -1, 1, 2]
    
    # Try products of two sparse polynomials with more trials
    for trial in range(3000):
        if time.time() > deadline:
            break
        # Choose sizes for A and B - favor smaller factors for more structure
        na = random.randint(2, 5)
        nb = random.randint(2, 6)
        if na * nb < 12:
            continue
        
        # Generate random exponents for A and B with some structure
        # Use arithmetic progressions or clustered exponents
        if random.random() < 0.5:
            # Arithmetic progression for A
            a_start = random.randint(0, 5)
            a_step = random.randint(1, 3)
            exps_a = [a_start + i * a_step for i in range(na)]
        else:
            exps_a = sorted(random.sample(range(0, 21), na))
        
        if random.random() < 0.5:
            # Arithmetic progression for B
            b_start = random.randint(0, 5)
            b_step = random.randint(1, 3)
            exps_b = [b_start + i * b_step for i in range(nb)]
        else:
            exps_b = sorted(random.sample(range(0, 21), nb))
        
        coeffs_a = [random.choice(coeff_choices) for _ in range(na)]
        coeffs_b = [random.choice(coeff_choices) for _ in range(nb)]
        
        # Compute product P = A * B
        prod = {}
        for i, ea in enumerate(exps_a):
            for j, eb in enumerate(exps_b):
                e = ea + eb
                prod[e] = prod.get(e, 0) + coeffs_a[i] * coeffs_b[j]
        
        # Filter to nonzero terms
        sparse = [(e, c) for e, c in sorted(prod.items()) if c != 0]
        
        # Check constraints
        if len(sparse) != 12:
            continue
        if sparse[-1][0] > 40:
            continue
        if any(abs(c) > 64 for _, c in sparse):
            continue
        
        try_candidate(sparse)
    
    # Phase 3: Enhanced local search with simulated annealing and targeted coefficient exploration
    if best_poly is not None:
        # Extract sparse representation from best_poly
        sparse_best = [(i, c) for i, c in enumerate(best_poly) if c != 0]
        
        # Simulated annealing with diverse mutation operators
        sa_temp = 10.0
        sa_cool = 0.995
        sa_best = list(sparse_best)
        sa_best_score = best_score
        sa_current = list(sparse_best)
        sa_current_score = best_score
        iterations = 0
        max_iterations = 25000
        
        while iterations < max_iterations and time.time() < deadline:
            iterations += 1
            
            # Create a mutated version
            mutated = list(sa_current)
            mutation_type = random.randint(0, 7)
            
            if mutation_type == 0 and len(mutated) > 2:
                # Change one coefficient (wider range)
                idx = random.randint(0, len(mutated) - 1)
                e, c = mutated[idx]
                new_c = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                if new_c != c:
                    mutated[idx] = (e, new_c)
            elif mutation_type == 1:
                # Move one exponent
                idx = random.randint(0, len(mutated) - 1)
                e, c = mutated[idx]
                new_e = e + random.randint(-6, 6)
                if 0 <= new_e <= 40:
                    if not any(ne == new_e for ne, nc in mutated if (ne, nc) != mutated[idx]):
                        mutated[idx] = (new_e, c)
            elif mutation_type == 2:
                # Swap two exponents
                if len(mutated) >= 2:
                    i, j = random.sample(range(len(mutated)), 2)
                    ei, ci = mutated[i]
                    ej, cj = mutated[j]
                    if ei != ej:
                        mutated[i] = (ej, ci)
                        mutated[j] = (ei, cj)
            elif mutation_type == 3:
                # Move two exponents (correlated)
                if len(mutated) >= 2:
                    i, j = random.sample(range(len(mutated)), 2)
                    ei, ci = mutated[i]
                    ej, cj = mutated[j]
                    delta = random.randint(-4, 4)
                    new_ei = ei + delta
                    new_ej = ej + delta
                    if 0 <= new_ei <= 40 and 0 <= new_ej <= 40:
                        if not any(ne == new_ei for ne, nc in mutated if (ne, nc) != mutated[i]):
                            if not any(ne == new_ej for ne, nc in mutated if (ne, nc) != mutated[j]):
                                mutated[i] = (new_ei, ci)
                                mutated[j] = (new_ej, cj)
            elif mutation_type == 4:
                # Sign flip on subset
                k = random.randint(1, 4)
                indices = random.sample(range(len(mutated)), k)
                for idx in indices:
                    e, c = mutated[idx]
                    mutated[idx] = (e, -c)
            elif mutation_type == 5:
                # Change two coefficients
                if len(mutated) >= 2:
                    i, j = random.sample(range(len(mutated)), 2)
                    for idx in [i, j]:
                        e, c = mutated[idx]
                        new_c = random.choice([-3, -2, -1, 1, 2, 3])
                        if new_c != c:
                            mutated[idx] = (e, new_c)
            elif mutation_type == 6:
                # Move a pair of exponents by same delta AND flip signs
                if len(mutated) >= 2:
                    i, j = random.sample(range(len(mutated)), 2)
                    ei, ci = mutated[i]
                    ej, cj = mutated[j]
                    delta = random.randint(-3, 3)
                    new_ei = ei + delta
                    new_ej = ej + delta
                    if 0 <= new_ei <= 40 and 0 <= new_ej <= 40:
                        if not any(ne == new_ei for ne, nc in mutated if (ne, nc) != mutated[i]):
                            if not any(ne == new_ej for ne, nc in mutated if (ne, nc) != mutated[j]):
                                mutated[i] = (new_ei, -ci)
                                mutated[j] = (new_ej, -cj)
            else:
                # Random restart with structure
                if random.random() < 0.4:
                    # AP-based
                    base = random.randint(0, 10)
                    step = random.randint(1, 4)
                    exps = [base + i * step for i in range(12)]
                    if exps[-1] <= 40:
                        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
                        mutated = [(exps[i], coeffs[i]) for i in range(12)]
                    else:
                        exps = sorted(random.sample(range(0, 41), 12))
                        coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
                        mutated = [(exps[i], coeffs[i]) for i in range(12)]
                else:
                    exps = sorted(random.sample(range(0, 41), 12))
                    coeffs = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
                    mutated = [(exps[i], coeffs[i]) for i in range(12)]
            
            # Validate mutated
            if len(mutated) != 12:
                continue
            if any(e < 0 or e > 40 for e, c in mutated):
                continue
            if any(abs(c) > 64 for e, c in mutated):
                continue
            exps_list = [e for e, c in mutated]
            if len(set(exps_list)) != 12:
                continue
            
            score = square_nonzero_count(mutated)
            
            # Simulated annealing acceptance
            if score <= sa_current_score:
                sa_current = mutated
                sa_current_score = score
            else:
                # Accept with probability exp(-delta/temp)
                if random.random() < math.exp(-(score - sa_current_score) / max(sa_temp, 0.01)):
                    sa_current = mutated
                    sa_current_score = score
            
            # Update best
            if score < sa_best_score:
                sa_best_score = score
                sa_best = mutated
            
            # Cool down
            sa_temp *= sa_cool
        
        # Update global best from SA
        if sa_best_score < best_score:
            best_score = sa_best_score
            poly = build_poly(sa_best)
            if poly is not None:
                best_poly = poly
                sparse_best = sa_best
    
    # Phase 4: Try palindromic structures
    for d_shift in range(0, 15):
        if time.time() > deadline:
            break
        base_exps = list(range(12))
        shifted_exps = [e + d_shift for e in base_exps]
        if shifted_exps[-1] > 40:
            continue
        for c0 in [-2, -1, 1, 2]:
            for c1 in [-2, -1, 1, 2]:
                for c2 in [-2, -1, 1, 2]:
                    for c3 in [-2, -1, 1, 2]:
                        for c4 in [-2, -1, 1, 2]:
                            for c5 in [-2, -1, 1, 2]:
                                coeffs_list = [c0, c1, c2, c3, c4, c5, c5, c4, c3, c2, c1, c0]
                                if sum(1 for c in coeffs_list if c != 0) != 12:
                                    continue
                                coeffs = [(shifted_exps[i], coeffs_list[i]) for i in range(12)]
                                try_candidate(coeffs)
    
    # Phase 5: Targeted search for low-square-count structures
    if time.time() < deadline:
        # Geometric-like progressions with small ratio
        for ratio in [2, 3]:
            for start in range(0, 5):
                exps = []
                e = start
                while len(exps) < 12 and e <= 40:
                    exps.append(e)
                    e *= ratio
                if len(exps) < 12:
                    continue
                exps = exps[:12]
                for signs_mask in range(1 << 12):
                    if time.time() > deadline:
                        break
                    coeffs = [(exps[i], 1 if (signs_mask >> i) & 1 else -1) for i in range(12)]
                    try_candidate(coeffs)
        
        # Clustered structures
        for cluster_start in range(0, 30):
            if time.time() > deadline:
                break
            for c1 in range(0, 4):
                for c2 in range(0, 4):
                    for c3 in range(0, 4):
                        exps = []
                        base = cluster_start
                        for c in [c1, c2, c3, c3, c2, c1]:
                            for k in range(3):
                                exps.append(base + c + k)
                        if len(exps) != 12 or exps[-1] > 40:
                            continue
                        for pattern in [[1,1,1,-1,-1,-1,1,1,1,-1,-1,-1],
                                        [1,-1,1,1,-1,1,-1,1,-1,1,-1,1],
                                        [1,1,-1,1,1,-1,-1,-1,1,-1,-1,1]]:
                            coeffs = [(exps[i], pattern[i]) for i in range(12)]
                            try_candidate(coeffs)
        
        # Phase 5b: Systematic coefficient space search on best exponent pattern
        if best_poly is not None and time.time() < deadline:
            sparse_best = [(i, c) for i, c in enumerate(best_poly) if c != 0]
            exps_best = [e for e, c in sparse_best]
            
            # Try many coefficient assignments on the best exponent pattern
            # Focus on coefficients that create cancellations
            coeff_options = [-2, -1, 1, 2]
            # Use a smarter enumeration: group by sign patterns
            for signs_mask in range(1 << 12):
                if time.time() > deadline:
                    break
                # Assign magnitudes based on position (smaller exponents get larger coeffs)
                for mag_pattern in range(1 << 6):  # 6 bits for 12 terms (pairs)
                    coeffs = []
                    for i in range(12):
                        sign = 1 if (signs_mask >> i) & 1 else -1
                        # Magnitude: use a hash of position and mask
                        mag_idx = (i // 2) & 0b11  # 0-3
                        mag = [1, 1, 2, 1][mag_idx]
                        coeffs.append((exps_best[i], sign * mag))
                    score = square_nonzero_count(coeffs)
                    if score < best_score:
                        best_score = score
                        poly = build_poly(coeffs)
                        if poly is not None:
                            best_poly = poly
    
    # Return best found, or a valid fallback
    if best_poly is not None:
        return best_poly
    
    return [1]*12

# EVOLVE-BLOCK-END
