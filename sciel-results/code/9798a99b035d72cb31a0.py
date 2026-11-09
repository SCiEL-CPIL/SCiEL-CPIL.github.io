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
    random.seed(42)
    coeff_choices = [-2, -1, 1, 2]
    
    for trial in range(3000):
        if time.time() > deadline:
            break
        na = random.randint(2, 5)
        nb = random.randint(2, 6)
        if na * nb < 12:
            continue
        
        if random.random() < 0.5:
            a_start = random.randint(0, 5)
            a_step = random.randint(1, 3)
            exps_a = [a_start + i * a_step for i in range(na)]
        else:
            exps_a = sorted(random.sample(range(0, 21), na))
        
        if random.random() < 0.5:
            b_start = random.randint(0, 5)
            b_step = random.randint(1, 3)
            exps_b = [b_start + i * b_step for i in range(nb)]
        else:
            exps_b = sorted(random.sample(range(0, 21), nb))
        
        coeffs_a = [random.choice(coeff_choices) for _ in range(na)]
        coeffs_b = [random.choice(coeff_choices) for _ in range(nb)]
        
        prod = {}
        for i, ea in enumerate(exps_a):
            for j, eb in enumerate(exps_b):
                e = ea + eb
                prod[e] = prod.get(e, 0) + coeffs_a[i] * coeffs_b[j]
        
        sparse = [(e, c) for e, c in sorted(prod.items()) if c != 0]
        
        if len(sparse) != 12:
            continue
        if sparse[-1][0] > 40:
            continue
        if any(abs(c) > 64 for _, c in sparse):
            continue
        
        try_candidate(sparse)
    
    # Phase 3: Simulated Annealing with structured initialization
    if best_poly is not None:
        sparse_best = [(i, c) for i, c in enumerate(best_poly) if c != 0]
        current_sparse = list(sparse_best)
        current_score = best_score
        
        # Temperature schedule
        T0 = 5.0
        T_end = 0.1
        total_iterations = 50000
        cooling_rate = (T_end / T0) ** (1.0 / total_iterations)
        T = T0
        
        for iteration in range(total_iterations):
            if time.time() > deadline:
                break
            
            # Generate mutation
            mutated = list(current_sparse)
            mutation_type = random.randint(0, 6)
            
            if mutation_type == 0:
                # Sign flip one term
                idx = random.randint(0, 11)
                e, c = mutated[idx]
                mutated[idx] = (e, -c)
            elif mutation_type == 1:
                # Sign flip two terms
                i, j = random.sample(range(12), 2)
                mutated[i] = (mutated[i][0], -mutated[i][1])
                mutated[j] = (mutated[j][0], -mutated[j][1])
            elif mutation_type == 2:
                # Move one exponent
                idx = random.randint(0, 11)
                e, c = mutated[idx]
                delta = random.randint(-3, 3)
                new_e = e + delta
                if 0 <= new_e <= 40:
                    if not any(ne == new_e for k, (ne, nc) in enumerate(mutated) if k != idx):
                        mutated[idx] = (new_e, c)
                    else:
                        mutated = list(current_sparse)  # revert
            elif mutation_type == 3:
                # Move two exponents by same delta
                i, j = random.sample(range(12), 2)
                delta = random.randint(-2, 2)
                new_ei = mutated[i][0] + delta
                new_ej = mutated[j][0] + delta
                if 0 <= new_ei <= 40 and 0 <= new_ej <= 40:
                    if not any(ne == new_ei for k, (ne, nc) in enumerate(mutated) if k not in (i, j)):
                        if not any(ne == new_ej for k, (ne, nc) in enumerate(mutated) if k not in (i, j)):
                            mutated[i] = (new_ei, mutated[i][1])
                            mutated[j] = (new_ej, mutated[j][1])
                        else:
                            mutated = list(current_sparse)
                    else:
                        mutated = list(current_sparse)
                else:
                    mutated = list(current_sparse)
            elif mutation_type == 4:
                # Change coefficient magnitude
                idx = random.randint(0, 11)
                e, c = mutated[idx]
                new_c = random.choice([-3, -2, -1, 1, 2, 3])
                if new_c != c:
                    mutated[idx] = (e, new_c)
            elif mutation_type == 5:
                # Swap two exponents
                i, j = random.sample(range(12), 2)
                ei, ci = mutated[i]
                ej, cj = mutated[j]
                if ei != ej:
                    mutated[i] = (ej, ci)
                    mutated[j] = (ei, cj)
            else:
                # Random restart with structure
                if random.random() < 0.4:
                    base = random.randint(0, 10)
                    step = random.randint(1, 3)
                    exps = [base + i * step for i in range(12)]
                    if exps[-1] <= 40:
                        coeffs = [random.choice(coeff_choices) for _ in range(12)]
                        mutated = [(exps[i], coeffs[i]) for i in range(12)]
                    else:
                        exps = sorted(random.sample(range(0, 41), 12))
                        coeffs = [random.choice(coeff_choices) for _ in range(12)]
                        mutated = [(exps[i], coeffs[i]) for i in range(12)]
                else:
                    exps = sorted(random.sample(range(0, 41), 12))
                    coeffs = [random.choice(coeff_choices) for _ in range(12)]
                    mutated = [(exps[i], coeffs[i]) for i in range(12)]
            
            # Validate
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
            
            # Acceptance criterion
            if score < current_score:
                # Always accept improvement
                current_sparse = mutated
                current_score = score
                if score < best_score:
                    best_score = score
                    poly = build_poly(mutated)
                    if poly is not None:
                        best_poly = poly
                        sparse_best = mutated
            else:
                # Accept with probability exp(-delta/T)
                delta = score - current_score
                if delta < 0:
                    delta = 0
                prob = math.exp(-delta / max(T, 0.001))
                if random.random() < prob:
                    current_sparse = mutated
                    current_score = score
            
            T *= cooling_rate
    
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
    
    if best_poly is not None:
        return best_poly
    
    return [1]*12

# EVOLVE-BLOCK-END
