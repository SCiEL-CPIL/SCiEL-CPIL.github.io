"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

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

    # Helper: check if we should stop
    def time_up():
        return time.time() > deadline

    # ============================================================
    # Phase 1: Exhaustive binary search on structured exponent sets
    # Key insight: For fixed exponent structure, ±1 coefficients
    # can create perfect cancellations in the square.
    # ============================================================

    # 1a: Arithmetic progressions
    for d in range(1, 5):
        max_a = 40 - 11 * d
        for a in range(0, max_a + 1):
            if time_up():
                if best_poly is not None:
                    return best_poly
                return [1]*12
            exponents = [a + i * d for i in range(12)]
            if exponents[-1] > 40:
                continue
            for signs_mask in range(1 << 12):
                if time_up():
                    if best_poly is not None:
                        return best_poly
                    return [1]*12
                coeffs = [(exponents[i], 1 if (signs_mask >> i) & 1 else -1) for i in range(12)]
                try_candidate(coeffs)

    # 1b: Symmetric exponent sets (palindromic structure)
    # exponents: a, a+d, a+2d, ..., a+5d, a+5d+d2, ..., a+11d (with possible gap)
    # Simpler: just try symmetric sets around a center
    for d in range(1, 5):
        for a in range(0, 20):
            if time_up():
                if best_poly is not None:
                    return best_poly
                return [1]*12
            # Symmetric: a+i*d for i=0..5, and a+(11-i)*d for i=0..5
            # But that gives duplicates. Instead: a, a+d, ..., a+5d, a+6d, ..., a+11d
            # which is just AP. Try: a, a+d, a+2d, a+3d, a+4d, a+5d, a+5d, a+4d... no.
            # Better: try sets with a gap in the middle
            # exponents = [a+i*d for i in range(6)] + [a+(11-i)*d for i in range(6)]
            # = [a, a+d, a+2d, a+3d, a+4d, a+5d, a+5d, a+4d, ...] -- duplicates!
            # Instead: [a+i*d for i in range(6)] + [a+(11-i)*d for i in range(6, 0, -1)]
            # No, let's just do: first 6 and last 6 with a gap
            # exponents = [a+i*d for i in range(6)] + [a+(6+gap)+i*d for i in range(6)]
            for gap in range(0, 4):
                exps = [a + i * d for i in range(6)] + [a + (6 + gap) + i * d for i in range(6)]
                if len(set(exps)) != 12 or max(exps) > 40:
                    continue
                exps = sorted(exps)
                for signs_mask in range(1 << 12):
                    if time_up():
                        if best_poly is not None:
                            return best_poly
                        return [1]*12
                    coeffs = [(exps[i], 1 if (signs_mask >> i) & 1 else -1) for i in range(12)]
                    try_candidate(coeffs)

    # 1c: Two-cluster structures: 6 terms in one AP, 6 in another
    for d1 in range(1, 4):
        for d2 in range(1, 4):
            for a1 in range(0, 15):
                for a2 in range(0, 25):
                    if time_up():
                        if best_poly is not None:
                            return best_poly
                        return [1]*12
                    cluster1 = [a1 + i * d1 for i in range(6)]
                    cluster2 = [a2 + i * d2 for i in range(6)]
                    all_exps = sorted(cluster1 + cluster2)
                    if len(set(all_exps)) != 12 or max(all_exps) > 40:
                        continue
                    for signs_mask in range(1 << 12):
                        if time_up():
                            if best_poly is not None:
                                return best_poly
                            return [1]*12
                        coeffs = [(all_exps[i], 1 if (signs_mask >> i) & 1 else -1) for i in range(12)]
                        try_candidate(coeffs)

    # ============================================================
    # Phase 2: Borwein-Erdelyi product construction
    # P = A * B where A and B are sparse. If A^2, B^2, and 2AB
    # have overlapping exponents with cancelling coefficients,
    # the square P^2 = A^2B^2 will be sparse.
    # ============================================================
    random.seed(42)
    coeff_choices = [-2, -1, 1, 2]

    # 2a: Exhaustive small product search
    # A has na terms, B has nb terms, na*nb >= 12 (but product may have fewer due to collisions)
    # Try na=2..4, nb=3..6, with small exponents
    for na in range(2, 5):
        for nb in range(3, 7):
            if na * nb < 12:
                continue
            # Generate all possible small exponent sets for A
            # A exponents from {0,1,2,3,4,5} choose na
            exps_a_list = list(itertools.combinations(range(0, 6), na))
            exps_b_list = list(itertools.combinations(range(0, 8), nb))

            for exps_a in exps_a_list:
                for exps_b in exps_b_list:
                    if time_up():
                        if best_poly is not None:
                            return best_poly
                        return [1]*12
                    # Try coefficient patterns
                    for coeffs_a in itertools.product([-1, 1], repeat=na):
                        for coeffs_b in itertools.product([-1, 1], repeat=nb):
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

    # 2b: Random product search with more coefficients
    for trial in range(3000):
        if time_up():
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

    # ============================================================
    # Phase 3: Local search / hill climbing
    # Refine the best found by mutating exponents and coefficients
    # ============================================================
    if best_poly is not None:
        sparse_best = [(i, c) for i, c in enumerate(best_poly) if c != 0]
        best_sparse = sparse_best
    else:
        # Fallback starting point
        best_sparse = [(i, 1) for i in range(12)]
        best_score = square_nonzero_count(best_sparse)
        best_poly = build_poly(best_sparse)

    # Multi-start local search
    for start_iter in range(3):
        if time_up():
            break

        if start_iter == 0:
            current_sparse = list(best_sparse)
        else:
            # Random restart
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice(coeff_choices) for _ in range(12)]
            current_sparse = [(exps[i], coeffs[i]) for i in range(12)]

        current_score = square_nonzero_count(current_sparse)

        # Hill climbing with simulated annealing-like acceptance
        for iteration in range(8000):
            if time_up():
                break

            # Create a mutated version
            mutated = list(current_sparse)
            mutation_type = random.randint(0, 4)

            if mutation_type == 0 and len(mutated) > 2:
                # Change a coefficient
                idx = random.randint(0, len(mutated) - 1)
                e, c = mutated[idx]
                new_c = random.choice([-2, -1, 1, 2])
                if new_c != c:
                    mutated[idx] = (e, new_c)
            elif mutation_type == 1:
                # Move an exponent by small amount
                idx = random.randint(0, len(mutated) - 1)
                e, c = mutated[idx]
                new_e = e + random.randint(-3, 3)
                if 0 <= new_e <= 40:
                    if not any(ne == new_e for k, (ne, nc) in enumerate(mutated) if k != idx):
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
                # Swap two coefficients
                if len(mutated) >= 2:
                    i, j = random.sample(range(len(mutated)), 2)
                    ei, ci = mutated[i]
                    ej, cj = mutated[j]
                    mutated[i] = (ei, cj)
                    mutated[j] = (ej, ci)
            else:
                # Random restart: new random sparse poly
                exps = sorted(random.sample(range(0, 41), 12))
                coeffs = [random.choice(coeff_choices) for _ in range(12)]
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

            # Accept if better, or occasionally accept worse (annealing)
            if score < current_score:
                current_score = score
                current_sparse = mutated
                if score < best_score:
                    best_score = score
                    poly = build_poly(mutated)
                    if poly is not None:
                        best_poly = poly
                        best_sparse = mutated
            elif score == current_score and random.random() < 0.3:
                # Accept equal to explore
                current_sparse = mutated

    # ============================================================
    # Phase 4: Targeted palindromic and symmetric structures
    # ============================================================
    for d_shift in range(0, 20):
        if time_up():
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

    # ============================================================
    # Phase 5: Final polish - try to reduce square terms by
    # adjusting coefficients of the best polynomial
    # ============================================================
    if best_poly is not None:
        sparse_final = [(i, c) for i, c in enumerate(best_poly) if c != 0]

        # Try all sign flips on the best structure
        for signs_mask in range(1 << 12):
            if time_up():
                break
            coeffs = [(sparse_final[i][0],
                      abs(sparse_final[i][1]) if (signs_mask >> i) & 1 else -abs(sparse_final[i][1]))
                      for i in range(12)]
            try_candidate(coeffs)

        # Try coefficient magnitude changes
        for i in range(12):
            if time_up():
                break
            e, c = sparse_final[i]
            for new_mag in [1, 2, 3, 4]:
                for sign in [-1, 1]:
                    new_c = sign * new_mag
                    if abs(new_c) > 64:
                        continue
                    mutated = list(sparse_final)
                    mutated[i] = (e, new_c)
                    try_candidate(mutated)

    # Return best found, or a valid fallback
    if best_poly is not None:
        return best_poly

    return [1]*12

# EVOLVE-BLOCK-END
