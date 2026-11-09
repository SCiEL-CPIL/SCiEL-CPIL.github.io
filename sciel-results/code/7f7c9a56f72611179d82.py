"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    best_poly = None
    best_score = float('inf')
    
    # Helper: compute sparse square coefficient count using numpy convolution
    def square_nonzero_count_numpy(sparse_coeffs):
        """Given list of (exponent, coefficient) pairs, return count of nonzero terms in square using numpy."""
        if not sparse_coeffs:
            return 0
        exps = np.array([e for e, c in sparse_coeffs], dtype=np.int64)
        coeffs = np.array([c for e, c in sparse_coeffs], dtype=np.int64)
        
        # Build dense array for numpy convolution
        max_exp = exps.max()
        dense = np.zeros(max_exp + 1, dtype=np.int64)
        dense[exps] = coeffs
        
        # Convolve with itself
        sq = np.convolve(dense, dense, mode='full')
        
        return int(np.count_nonzero(sq))
    
    # Fallback pure-python version (for very small inputs where numpy overhead dominates)
    def square_nonzero_count(coeffs):
        """Given list of (exponent, coefficient) pairs, return count of nonzero terms in square."""
        n = len(coeffs)
        if n == 0:
            return 0
        if n <= 4:
            # Use pure python for small inputs
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
            return sum(1 for v in sq.values() if v != 0)
        else:
            return square_nonzero_count_numpy(coeffs)
    
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
    
    # Try products of two sparse polynomials
    for trial in range(2000):
        if time.time() > deadline:
            break
        na = random.randint(2, 6)
        nb = random.randint(2, 6)
        if na * nb < 12:
            continue
        
        exps_a = sorted(random.sample(range(0, 21), na))
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
    
    # Phase 3: Aggressive local search with numpy-accelerated scoring
    if best_poly is not None:
        sparse_best = [(i, c) for i, c in enumerate(best_poly) if c != 0]
        
        # Precompute for fast numpy scoring
        best_exps = np.array([e for e, c in sparse_best], dtype=np.int64)
        best_coeffs_arr = np.array([c for e, c in sparse_best], dtype=np.int64)
        max_exp_global = int(best_exps.max())
        
        def fast_score_from_arrays(exps_arr, coeffs_arr):
            """Fast scoring using pre-allocated numpy arrays."""
            if len(exps_arr) == 0:
                return 0
            max_e = int(exps_arr.max())
            if max_e > max_exp_global:
                dense = np.zeros(max_e + 1, dtype=np.int64)
            else:
                dense = np.zeros(max_exp_global + 1, dtype=np.int64)
            dense[exps_arr] = coeffs_arr
            sq = np.convolve(dense, dense, mode='full')
            return int(np.count_nonzero(sq))
        
        # Extended local search with more iterations
        search_budget = int((deadline - time.time()) * 0.6)  # 60% of remaining time
        search_deadline = time.time() + search_budget
        
        for iteration in range(50000):
            if time.time() > search_deadline:
                break
            
            mutated = list(sparse_best)
            mutation_type = random.randint(0, 4)
            
            if mutation_type == 0:
                # Change a coefficient
                idx = random.randint(0, len(mutated) - 1)
                e, c = mutated[idx]
                new_c = random.choice([-3, -2, -1, 1, 2, 3])
                if new_c != c:
                    mutated[idx] = (e, new_c)
            elif mutation_type == 1:
                # Move an exponent
                idx = random.randint(0, len(mutated) - 1)
                e, c = mutated[idx]
                new_e = e + random.randint(-4, 4)
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
                # Change two coefficients simultaneously
                if len(mutated) >= 2:
                    i, j = random.sample(range(len(mutated)), 2)
                    ei, ci = mutated[i]
                    ej, cj = mutated[j]
                    new_ci = random.choice([-3, -2, -1, 1, 2, 3])
                    new_cj = random.choice([-3, -2, -1, 1, 2, 3])
                    if new_ci != ci:
                        mutated[i] = (ei, new_ci)
                    if new_cj != cj:
                        mutated[j] = (ej, new_cj)
            else:
                # Random restart
                exps = sorted(random.sample(range(0, 41), 12))
                coeffs = [random.choice(coeff_choices) for _ in range(12)]
                mutated = [(exps[i], coeffs[i]) for i in range(12)]
            
            if len(mutated) != 12:
                continue
            if any(e < 0 or e > 40 for e, c in mutated):
                continue
            if any(abs(c) > 64 for e, c in mutated):
                continue
            exps_list = [e for e, c in mutated]
            if len(set(exps_list)) != 12:
                continue
            
            # Fast numpy scoring
            m_exps = np.array(exps_list, dtype=np.int64)
            m_coeffs = np.array([c for e, c in mutated], dtype=np.int64)
            score = fast_score_from_arrays(m_exps, m_coeffs)
            
            if score < best_score:
                best_score = score
                poly = build_poly(mutated)
                if poly is not None:
                    best_poly = poly
                    sparse_best = mutated
                    best_exps = m_exps.copy()
                    best_coeffs_arr = m_coeffs.copy()
                    max_exp_global = int(best_exps.max())
    
    # Phase 4: Palindromic and symmetric structures
    for d_shift in range(0, 20):
        if time.time() > deadline:
            break
        base_exps = list(range(12))
        shifted_exps = [e + d_shift for e in base_exps]
        if shifted_exps[-1] > 40:
            continue
        for c0 in [-3, -2, -1, 1, 2, 3]:
            for c1 in [-3, -2, -1, 1, 2, 3]:
                for c2 in [-3, -2, -1, 1, 2, 3]:
                    for c3 in [-3, -2, -1, 1, 2, 3]:
                        for c4 in [-3, -2, -1, 1, 2, 3]:
                            for c5 in [-3, -2, -1, 1, 2, 3]:
                                coeffs_list = [c0, c1, c2, c3, c4, c5, c5, c4, c3, c2, c1, c0]
                                if sum(1 for c in coeffs_list if c != 0) != 12:
                                    continue
                                coeffs = [(shifted_exps[i], coeffs_list[i]) for i in range(12)]
                                try_candidate(coeffs)
    
    # Phase 5: Final polish - targeted local search around best
    if best_poly is not None and time.time() < deadline - 5:
        sparse_best = [(i, c) for i, c in enumerate(best_poly) if c != 0]
        polish_deadline = deadline - 2
        
        for iteration in range(20000):
            if time.time() > polish_deadline:
                break
            
            mutated = list(sparse_best)
            # Focus on single-exponent moves and coefficient tweaks
            if random.random() < 0.5:
                idx = random.randint(0, len(mutated) - 1)
                e, c = mutated[idx]
                new_e = e + random.randint(-2, 2)
                if 0 <= new_e <= 40 and not any(ne == new_e for ne, nc in mutated if (ne, nc) != mutated[idx]):
                    mutated[idx] = (new_e, c)
            else:
                idx = random.randint(0, len(mutated) - 1)
                e, c = mutated[idx]
                new_c = random.choice([-3, -2, -1, 1, 2, 3])
                if new_c != c:
                    mutated[idx] = (e, new_c)
            
            if len(mutated) != 12:
                continue
            if any(e < 0 or e > 40 for e, c in mutated):
                continue
            if any(abs(c) > 64 for e, c in mutated):
                continue
            exps_list = [e for e, c in mutated]
            if len(set(exps_list)) != 12:
                continue
            
            m_exps = np.array(exps_list, dtype=np.int64)
            m_coeffs = np.array([c for e, c in mutated], dtype=np.int64)
            max_e = int(m_exps.max())
            dense = np.zeros(max_e + 1, dtype=np.int64)
            dense[m_exps] = m_coeffs
            sq = np.convolve(dense, dense, mode='full')
            score = int(np.count_nonzero(sq))
            
            if score < best_score:
                best_score = score
                poly = build_poly(mutated)
                if poly is not None:
                    best_poly = poly
                    sparse_best = mutated
    
    if best_poly is not None:
        return best_poly
    
    return [1]*12

# EVOLVE-BLOCK-END
