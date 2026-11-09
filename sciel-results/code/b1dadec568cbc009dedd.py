"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill

    best_poly = None
    best_score = float('inf')

    # Helper: compute sparse square coefficient count (optimized)
    def square_nonzero_count(coeffs):
        """Given list of (exponent, coefficient) pairs, return count of nonzero terms in square."""
        n = len(coeffs)
        if n == 0:
            return 0
        sq = {}
        for i in range(n):
            ei, ci = coeffs[i]
            sq[2*ei] = sq.get(2*ei, 0) + ci * ci
            for j in range(i+1, n):
                ej, cj = coeffs[j]
                exp = ei + ej
                sq[exp] = sq.get(exp, 0) + 2 * ci * cj
        return sum(1 for v in sq.values() if v != 0)

    # Helper: validate and trim
    def validate_and_trim(poly):
        if not poly:
            return None
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

    # Phase 1: Exhaustive AP search with ±1 coefficients (fast)
    for d in range(1, 5):
        max_a = 40 - 11 * d
        if max_a < 0:
            continue
        for a in range(0, max_a + 1):
            exponents = [a + i * d for i in range(12)]
            for signs_mask in range(1 << 12):
                if time.time() > deadline:
                    if best_poly is not None:
                        return best_poly
                    return [1]*12
                coeffs = [(exponents[i], 1 if (signs_mask >> i) & 1 else -1) for i in range(12)]
                try_candidate(coeffs)

    # Phase 2: Product construction P = A * B
    random.seed(42)
    coeff_choices = [-3, -2, -1, 1, 2, 3]

    for trial in range(5000):
        if time.time() > deadline:
            break
        na = random.randint(2, 6)
        nb = 12 - na
        if nb < 1:
            continue

        # Generate exponent sets for A and B
        a_start = random.randint(0, 8)
        a_step = random.randint(1, 3)
        exps_a = [a_start + i * a_step for i in range(na)]

        b_start = random.randint(0, 8)
        b_step = random.randint(1, 3)
        exps_b = [b_start + i * b_step for i in range(nb)]

        coeffs_a = [random.choice(coeff_choices) for _ in range(na)]
        coeffs_b = [random.choice(coeff_choices) for _ in range(nb)]

        # Compute product
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

    # Phase 3: Cancellation-basis composition
    # Pre-compute small polynomials with sparse squares, then compose them
    if time.time() < deadline:
        # Find small 3- and 4-term polynomials with sparse squares
        small_bases = []
        for n_terms in [3, 4]:
            for exps in itertools.combinations(range(0, 15), n_terms):
                if time.time() > deadline:
                    break
                for signs in itertools.product([-1, 1], repeat=n_terms):
                    coeffs = list(zip(exps, signs))
                    sq_count = square_nonzero_count(coeffs)
                    if sq_count <= n_terms * 2:  # sparse square
                        small_bases.append(coeffs)

        # Compose bases: P = B1 + shift(B2) to get 12 terms
        for b1 in small_bases:
            if time.time() > deadline:
                break
            for b2 in small_bases:
                if len(b1) + len(b2) != 12:
                    continue
                if time.time() > deadline:
                    break
                # Try various shifts for b2
                for shift in range(0, 25):
                    if time.time() > deadline:
                        break
                    shifted_b2 = [(e + shift, c) for e, c in b2]
                    combined = b1 + shifted_b2
                    if len(combined) != 12:
                        continue
                    if any(e > 40 for e, c in combined):
                        continue
                    if any(abs(c) > 64 for e, c in combined):
                        continue
                    if len(set(e for e, c in combined)) != 12:
                        continue
                    try_candidate(combined)

    # Phase 4: Simulated Annealing with structured mutations
    if best_poly is not None:
        sparse_best = [(i, c) for i, c in enumerate(best_poly) if c != 0]
        current_sparse = list(sparse_best)
        current_score = best_score

        T0 = 5.0
        T_end = 0.1
        total_iterations = 120000
        cooling_rate = (T_end / T0) ** (1.0 / total_iterations)
        T = T0

        for iteration in range(total_iterations):
            if time.time() > deadline:
                break

            mutated = list(current_sparse)
            mutation_type = random.randint(0, 9)

            if mutation_type == 0:
                idx = random.randint(0, 11)
                e, c = mutated[idx]
                mutated[idx] = (e, -c)
            elif mutation_type == 1:
                i, j = random.sample(range(12), 2)
                mutated[i] = (mutated[i][0], -mutated[i][1])
                mutated[j] = (mutated[j][0], -mutated[j][1])
            elif mutation_type == 2:
                idx = random.randint(0, 11)
                e, c = mutated[idx]
                delta = random.randint(-4, 4)
                new_e = e + delta
                if 0 <= new_e <= 40:
                    if not any(ne == new_e for k, (ne, nc) in enumerate(mutated) if k != idx):
                        mutated[idx] = (new_e, c)
                    else:
                        mutated = list(current_sparse)
            elif mutation_type == 3:
                i, j = random.sample(range(12), 2)
                delta = random.randint(-3, 3)
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
                idx = random.randint(0, 11)
                e, c = mutated[idx]
                new_c = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                if new_c != c and abs(new_c) <= 64:
                    mutated[idx] = (e, new_c)
            elif mutation_type == 5:
                i, j = random.sample(range(12), 2)
                ei, ci = mutated[i]
                ej, cj = mutated[j]
                if ei != ej:
                    mutated[i] = (ej, ci)
                    mutated[j] = (ei, cj)
            elif mutation_type == 6:
                # Move 3 exponents by same delta (preserves structure)
                i, j, k = random.sample(range(12), 3)
                delta = random.randint(-2, 2)
                new_ei = mutated[i][0] + delta
                new_ej = mutated[j][0] + delta
                new_ek = mutated[k][0] + delta
                if 0 <= new_ei <= 40 and 0 <= new_ej <= 40 and 0 <= new_ek <= 40:
                    other_exps = set(mutated[m][0] for m in range(12) if m not in (i, j, k))
                    if new_ei not in other_exps and new_ej not in other_exps and new_ek not in other_exps:
                        mutated[i] = (new_ei, mutated[i][1])
                        mutated[j] = (new_ej, mutated[j][1])
                        mutated[k] = (new_ek, mutated[k][1])
                    else:
                        mutated = list(current_sparse)
                else:
                    mutated = list(current_sparse)
            elif mutation_type == 7:
                # Swap coefficients between two terms
                i, j = random.sample(range(12), 2)
                ei, ci = mutated[i]
                ej, cj = mutated[j]
                if ci != cj:
                    mutated[i] = (ei, cj)
                    mutated[j] = (ej, ci)
            elif mutation_type == 8:
                # Shift a contiguous block of exponents
                block_size = random.randint(2, 4)
                start_idx = random.randint(0, 12 - block_size)
                delta = random.randint(-3, 3)
                new_exps = list(mutated[k][0] for k in range(12))
                for k in range(start_idx, start_idx + block_size):
                    new_exps[k] += delta
                if all(0 <= e <= 40 for e in new_exps) and len(set(new_exps)) == 12:
                    mutated = [(new_exps[k], mutated[k][1]) for k in range(12)]
                else:
                    mutated = list(current_sparse)
            else:
                # Random restart with AP structure
                if random.random() < 0.5:
                    base = random.randint(0, 10)
                    step = random.randint(1, 4)
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

            if score < current_score:
                current_sparse = mutated
                current_score = score
                if score < best_score:
                    best_score = score
                    poly = build_poly(mutated)
                    if poly is not None:
                        best_poly = poly
                        sparse_best = mutated
            else:
                delta = score - current_score
                prob = math.exp(-delta / max(T, 0.001))
                if random.random() < prob:
                    current_sparse = mutated
                    current_score = score

            T *= cooling_rate

    # Phase 5: Palindromic structures
    for d_shift in range(0, 15):
        if time.time() > deadline:
            break
        shifted_exps = [e + d_shift for e in range(12)]
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

    # Phase 6: Targeted geometric progression and cluster searches
    if time.time() < deadline:
        for ratio in [2, 3]:
            for start in range(0, 6):
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
            for gap1 in range(0, 5):
                for gap2 in range(0, 5):
                    for gap3 in range(0, 5):
                        exps = []
                        base = cluster_start
                        for c in [0, gap1, gap1+gap2, gap1+gap2+gap3]:
                            for k in range(3):
                                exps.append(base + c + k)
                        if len(exps) != 12 or exps[-1] > 40:
                            continue
                        if len(set(exps)) != 12:
                            continue
                        for pattern in [[1]*12, [1,1,1,-1,-1,-1,1,1,1,-1,-1,-1],
                                        [1,-1,1,1,-1,1,-1,1,-1,1,-1,1]]:
                            coeffs = [(exps[i], pattern[i]) for i in range(12)]
                            try_candidate(coeffs)

    # Phase 7: Final local-search polish on best candidate
    if best_poly is not None and time.time() < deadline - 2:
        sparse_best = [(i, c) for i, c in enumerate(best_poly) if c != 0]
        current_sparse = list(sparse_best)
        current_score = best_score

        # Optimized local search with multiple passes
        max_passes = 10
        for _pass in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            for idx in range(12):
                if time.time() > deadline:
                    break
                e, c = current_sparse[idx]
                # Try sign flip
                trial = list(current_sparse)
                trial[idx] = (e, -c)
                s = square_nonzero_count(trial)
                if s < current_score:
                    current_sparse = trial
                    current_score = s
                    improved = True
                    if s < best_score:
                        best_score = s
                        poly = build_poly(current_sparse)
                        if poly is not None:
                            best_poly = poly
                # Try coefficient changes (wider range)
                for nc in [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]:
                    if nc == c or abs(nc) > 64:
                        continue
                    trial = list(current_sparse)
                    trial[idx] = (e, nc)
                    s = square_nonzero_count(trial)
                    if s < current_score:
                        current_sparse = trial
                        current_score = s
                        improved = True
                        if s < best_score:
                            best_score = s
                            poly = build_poly(current_sparse)
                            if poly is not None:
                                best_poly = poly
                # Try exponent shifts (wider range)
                for de in range(-5, 6):
                    if de == 0:
                        continue
                    ne = e + de
                    if ne < 0 or ne > 40:
                        continue
                    if any(other[0] == ne for k, other in enumerate(current_sparse) if k != idx):
                        continue
                    trial = list(current_sparse)
                    trial[idx] = (ne, c)
                    s = square_nonzero_count(trial)
                    if s < current_score:
                        current_sparse = trial
                        current_score = s
                        improved = True
                        if s < best_score:
                            best_score = s
                            poly = build_poly(current_sparse)
                            if poly is not None:
                                best_poly = poly
            if not improved:
                break

    if best_poly is not None:
        return best_poly

    return [1]*12

# EVOLVE-BLOCK-END
