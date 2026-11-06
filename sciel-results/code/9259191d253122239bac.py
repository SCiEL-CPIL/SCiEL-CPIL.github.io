"""Sparse squares of 12-term integer polynomials – stronger search."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill

    best_poly = None
    best_nz = float('inf')

    def square_nonzero(exponents, coeffs):
        """Fast exact square nonzero count via numpy."""
        c = np.asarray(coeffs, dtype=np.int64)
        e = np.asarray(exponents, dtype=np.int64)
        degs = e[:, None] + e[None, :]
        vals = c[:, None] * c[None, :]
        result = np.bincount(degs.ravel(), weights=vals.ravel(), minlength=82)
        return int(np.count_nonzero(result))

    def square_nonzero_fast(exponents, coeffs):
        """Even faster: precompute once, use for many coefficient tests on same exponents."""
        e = np.asarray(exponents, dtype=np.int64)
        degs = e[:, None] + e[None, :]
        # Precompute the degree matrix structure
        return degs

    def validate(poly_coeff_list):
        if not poly_coeff_list or len(poly_coeff_list) > 41:
            return False
        if poly_coeff_list[-1] == 0:
            return False
        for c in poly_coeff_list:
            if not isinstance(c, int) or c < -64 or c > 64:
                return False
        if sum(1 for c in poly_coeff_list if c != 0) != 12:
            return False
        return True

    def try_candidate(exponents, coeffs):
        nonlocal best_poly, best_nz
        if len(exponents) != 12 or len(set(exponents)) != 12:
            return
        if any(e < 0 or e >= 41 for e in exponents):
            return
        if any(c == 0 or abs(c) > 64 for c in coeffs):
            return
        nz = square_nonzero(exponents, coeffs)
        if nz < best_nz:
            best_nz = nz
            max_deg = max(exponents)
            poly = [0] * (max_deg + 1)
            for e, c in zip(exponents, coeffs):
                poly[e] = c
            best_poly = poly

    # ---------- Phase 1: Exhaustive AP exponents with rich coefficient patterns ----------
    # Build a larger set of coefficient patterns
    sign_patterns = []
    for period in [1, 2, 3, 4, 6, 12]:
        for phase in range(period):
            pat = [1 if ((i + phase) % (2 * period)) < period else -1 for i in range(12)]
            sign_patterns.append(pat)

    extra_patterns = [
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
        [1, 2, -1, -2, 1, 2, -1, -2, 1, 2, -1, -2],
        [1, -2, 2, -1, 1, -2, 2, -1, 1, -2, 2, -1],
        [2, -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2],
        [1, 1, -2, -1, 1, 1, -2, -1, 1, 1, -2, -1],
        [3, 1, -1, -3, -1, 1, 3, 1, -1, -3, -1, 1],
        [1, 3, -1, -3, 1, 3, -1, -3, 1, 3, -1, -3],
        [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1],
        [1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1],
        [1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1],
        [1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1],
        [2, 2, -2, -2, 2, 2, -2, -2, 2, 2, -2, -2],
        [1, 2, 3, -1, -2, -3, 1, 2, 3, -1, -2, -3],
        [1, 2, -3, -1, -2, 3, 1, 2, -3, -1, -2, 3],
        [3, -2, 1, -3, 2, -1, 3, -2, 1, -3, 2, -1],
        [1, -3, 2, 1, -3, 2, 1, -3, 2, 1, -3, 2],
        [2, -3, 1, 2, -3, 1, 2, -3, 1, 2, -3, 1],
    ]
    all_patterns = sign_patterns + extra_patterns

    for d in range(1, 8):
        if time.time() > deadline - 35:
            break
        max_a = min(40 - 11 * d, 20)
        for a in range(0, max_a + 1):
            if time.time() > deadline - 35:
                break
            exponents = [a + i * d for i in range(12)]
            if exponents[-1] >= 41:
                continue
            for cp in all_patterns:
                if time.time() > deadline - 35:
                    break
                try_candidate(exponents, cp)

    # ---------- Phase 2: Two-AP union structures (6+6) ----------
    if time.time() < deadline - 30:
        for d1 in range(1, 6):
            for d2 in range(1, 6):
                if time.time() > deadline - 28:
                    break
                for a1 in range(0, 14):
                    for gap in range(3, 25):
                        if time.time() > deadline - 28:
                            break
                        a2 = a1 + gap
                        exp1 = [a1 + i * d1 for i in range(6)]
                        exp2 = [a2 + i * d2 for i in range(6)]
                        exponents = sorted(exp1 + exp2)
                        if len(set(exponents)) != 12 or any(e >= 41 for e in exponents):
                            continue
                        for s1 in [1, -1]:
                            for s2 in [1, -1]:
                                if time.time() > deadline - 28:
                                    break
                                coeffs = [s1] * 6 + [s2] * 6
                                try_candidate(exponents, coeffs)
                                c_alt = [s1 * (1 if i % 2 == 0 else -1) for i in range(6)] + \
                                        [s2 * (1 if i % 2 == 0 else -1) for i in range(6)]
                                try_candidate(exponents, c_alt)
                                c_blk = [s1 * (1 if (i // 2) % 2 == 0 else -1) for i in range(6)] + \
                                        [s2 * (1 if (i // 2) % 2 == 0 else -1) for i in range(6)]
                                try_candidate(exponents, c_blk)
                                # Also try mixing magnitudes
                                c_mix = [s1 * (1 + (i % 2)) for i in range(6)] + \
                                        [s2 * (1 + (i % 2)) for i in range(6)]
                                try_candidate(exponents, c_mix)

    # ---------- Phase 3: Symmetric exponent structures ----------
    if time.time() < deadline - 25:
        offset_sets = [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 7],
            [1, 2, 3, 4, 6, 8],
            [1, 2, 3, 5, 7, 9],
            [1, 3, 5, 7, 9, 11],
            [2, 4, 6, 8, 10, 12],
            [1, 2, 4, 5, 7, 8],
            [1, 3, 4, 6, 7, 9],
            [1, 2, 4, 6, 8, 10],
            [2, 3, 4, 5, 6, 7],
            [1, 2, 3, 5, 6, 8],
            [1, 3, 4, 5, 7, 9],
            [1, 2, 4, 5, 6, 8],
            [2, 3, 5, 6, 8, 10],
            [1, 2, 3, 4, 6, 7],
        ]
        for center in range(6, 35):
            if time.time() > deadline - 25:
                break
            for offsets in offset_sets:
                if time.time() > deadline - 25:
                    break
                exponents = sorted([center - o for o in offsets] + [center + o for o in offsets])
                if any(e < 0 or e >= 41 for e in exponents) or len(set(exponents)) != 12:
                    continue
                for cp in all_patterns[:25]:
                    if time.time() > deadline - 25:
                        break
                    try_candidate(exponents, cp)

    # ---------- Phase 4: Randomized search with diverse strategies ----------
    random.seed(42)

    # Extended coefficient patterns for random search
    extended_patterns = all_patterns + [
        [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1],
        [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
        [1, 2, 1, -1, -2, -1, 1, 2, 1, -1, -2, -1],
        [2, 1, -1, -2, 1, -1, 2, 1, -1, -2, 1, -1],
        [1, -2, 2, -1, -1, 2, -2, 1, 1, -2, 2, -1],
        [3, -1, -1, 3, 1, 1, -3, 1, 1, -3, -1, -1],
        [1, 3, 1, -1, -3, -1, 1, 3, 1, -1, -3, -1],
        [1, 1, 2, -1, -1, -2, 1, 1, 2, -1, -1, -2],
        [2, -1, 1, -2, 1, -2, 2, -1, 1, -2, 1, -2],
        [1, -1, 2, -2, 2, -2, 1, -1, 2, -2, 2, -2],
        [1, 2, -2, 1, 1, 2, -2, 1, 1, 2, -2, 1],
    ]

    def random_search(num_trials=200000):
        nonlocal best_poly, best_nz
        for trial in range(num_trials):
            if time.time() > deadline - 20:
                break

            r = random.random()
            if r < 0.18:
                d = random.randint(1, 6)
                max_a = max(0, 40 - 11 * d)
                a = random.randint(0, max_a)
                exponents = [a + i * d for i in range(12)]
                if exponents[-1] >= 41:
                    continue
            elif r < 0.33:
                d1 = random.randint(1, 5)
                d2 = random.randint(1, 5)
                a1 = random.randint(0, 12)
                gap = random.randint(4, 20)
                a2 = a1 + gap
                exponents = [a1 + i * d1 for i in range(6)] + [a2 + i * d2 for i in range(6)]
                if any(e >= 41 for e in exponents) or len(set(exponents)) != 12:
                    continue
                exponents.sort()
            elif r < 0.48:
                center = random.randint(6, 34)
                offsets = sorted(random.sample(range(1, 20), 6))
                exponents = [center - o for o in offsets] + [center + o for o in offsets]
                if any(e < 0 or e >= 41 for e in exponents) or len(set(exponents)) != 12:
                    continue
                exponents.sort()
            elif r < 0.63:
                window = random.randint(14, 28)
                base = random.randint(0, max(0, 40 - window))
                exponents = sorted(random.sample(range(base, min(base + window, 41)), 12))
            elif r < 0.78:
                d1 = random.randint(1, 3)
                d2 = random.randint(1, 3)
                d3 = random.randint(1, 3)
                a1 = random.randint(0, 5)
                g1 = random.randint(3, 8)
                a2 = a1 + g1
                g2 = random.randint(3, 8)
                a3 = a2 + g2
                exponents = ([a1 + i * d1 for i in range(3)] +
                             [a2 + i * d2 for i in range(4)] +
                             [a3 + i * d3 for i in range(5)])
                if any(e >= 41 for e in exponents) or len(set(exponents)) != 12:
                    continue
                exponents.sort()
            else:
                exponents = sorted(random.sample(range(41), 12))

            cstyle = random.random()
            if cstyle < 0.20:
                coeffs = [random.choice([-1, 1]) for _ in range(12)]
            elif cstyle < 0.35:
                coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            elif cstyle < 0.55:
                pat = random.choice(extended_patterns)
                coeffs = list(pat)
            elif cstyle < 0.70:
                coeffs = [random.randint(-4, 4) for _ in range(12)]
                for i in range(12):
                    while coeffs[i] == 0:
                        coeffs[i] = random.randint(1, 4) * random.choice([-1, 1])
            elif cstyle < 0.85:
                coeffs = [random.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for _ in range(12)]
            else:
                # Larger coefficient magnitudes
                coeffs = [random.choice([-6, -4, -3, -2, -1, 1, 2, 3, 4, 6]) for _ in range(12)]

            if all(abs(c) <= 64 and c != 0 for c in coeffs):
                try_candidate(exponents, coeffs)

    random_search(200000)

    # ---------- Phase 5: Product of two 6-term polynomials ----------
    if time.time() < deadline - 25:
        sign_pats_6 = []
        for period in [1, 2, 3, 6]:
            for phase in range(period):
                pat = [1 if ((i + phase) % (2 * period)) < period else -1 for i in range(6)]
                sign_pats_6.append(pat)
        extra6 = [
            [1, 1, -1, -1, 1, 1],
            [1, -1, 1, -1, 1, -1],
            [1, 1, 1, -1, -1, -1],
            [1, -1, -1, 1, 1, -1],
            [1, 1, -2, -1, 1, 1],
            [1, -2, 2, -1, 1, -2],
            [2, -1, 1, -2, 2, -1],
            [1, 2, -1, -2, 1, 2],
            [3, 1, -1, -3, -1, 1],
            [1, 3, -1, -3, 1, 3],
            [1, 1, -1, -1, -1, -1],
            [1, -1, -1, 1, -1, 1],
            [-1, 1, 1, -1, 1, -1],
            [1, -1, 1, 1, -1, -1],
            [2, 1, -1, -2, 1, 2],
            [1, 2, -2, -1, 2, -2],
            [2, -1, -1, 2, -1, 1],
            [1, -2, -2, 1, 2, 2],
        ]
        all_pats_6 = sign_pats_6 + extra6

        for d1 in range(1, 6):
            for d2 in range(1, 6):
                if time.time() > deadline - 20:
                    break
                for a1 in range(0, 12):
                    for a2 in range(0, 12):
                        if time.time() > deadline - 20:
                            break
                        q_exp = [a1 + i * d1 for i in range(6)]
                        r_exp = [a2 + i * d2 for i in range(6)]
                        if len(set(q_exp + r_exp)) != 12:
                            continue
                        if any(e >= 41 for e in q_exp + r_exp):
                            continue
                        for pq in all_pats_6:
                            for pr in all_pats_6:
                                if time.time() > deadline - 20:
                                    break
                                for s1 in [1, -1]:
                                    for s2 in [1, -1]:
                                        q_coef = [s1 * c for c in pq]
                                        r_coef = [s2 * c for c in pr]
                                        exponents = sorted(q_exp + r_exp)
                                        q_map = dict(zip(q_exp, q_coef))
                                        r_map = dict(zip(r_exp, r_coef))
                                        coeffs = []
                                        for e in exponents:
                                            if e in q_map:
                                                coeffs.append(q_map[e])
                                            else:
                                                coeffs.append(r_map[e])
                                        try_candidate(exponents, coeffs)

        # Randomized product search
        for trial in range(50000):
            if time.time() > deadline - 15:
                break
            d1 = random.randint(1, 5)
            d2 = random.randint(1, 5)
            a1 = random.randint(0, 14)
            a2 = random.randint(0, 14)
            q_exp = [a1 + i * d1 for i in range(6)]
            r_exp = [a2 + i * d2 for i in range(6)]
            if len(set(q_exp + r_exp)) != 12:
                continue
            if any(e >= 41 for e in q_exp + r_exp):
                continue
            style = random.random()
            if style < 0.4:
                q_coef = [random.choice([-1, 1]) for _ in range(6)]
                r_coef = [random.choice([-1, 1]) for _ in range(6)]
            elif style < 0.6:
                q_coef = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
                r_coef = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
            elif style < 0.8:
                q_coef = list(random.choice(all_pats_6))
                r_coef = list(random.choice(all_pats_6))
            else:
                q_coef = [random.randint(-3, 3) for _ in range(6)]
                r_coef = [random.randint(-3, 3) for _ in range(6)]
                for i in range(6):
                    while q_coef[i] == 0:
                        q_coef[i] = random.randint(1, 3) * random.choice([-1, 1])
                    while r_coef[i] == 0:
                        r_coef[i] = random.randint(1, 3) * random.choice([-1, 1])
            if any(abs(c) > 64 or c == 0 for c in q_coef + r_coef):
                continue
            exponents = sorted(q_exp + r_exp)
            q_map = dict(zip(q_exp, q_coef))
            r_map = dict(zip(r_exp, r_coef))
            coeffs = []
            for e in exponents:
                if e in q_map:
                    coeffs.append(q_map[e])
                else:
                    coeffs.append(r_map[e])
            try_candidate(exponents, coeffs)

    # ---------- Phase 6: Enhanced local search with coefficient-focused optimization ----------
    if best_poly is not None and time.time() < deadline - 5:
        best_exp = [i for i, c in enumerate(best_poly) if c != 0]
        best_coef = [c for c in best_poly if c != 0]

        # Precompute degree matrix for current exponents for fast coefficient-only eval
        def eval_coefs_only(exp_list, coef_list):
            """Evaluate square nonzero count for given exponents and coefficients."""
            c = np.asarray(coef_list, dtype=np.int64)
            e = np.asarray(exp_list, dtype=np.int64)
            degs = e[:, None] + e[None, :]
            vals = c[:, None] * c[None, :]
            result = np.bincount(degs.ravel(), weights=vals.ravel(), minlength=82)
            return int(np.count_nonzero(result))

        # Coefficient value pool for local search
        coef_pool = list(range(-8, 9)) + [10, 11, 12, 14, 16, 20, 24, 32, 48, 64]
        coef_pool_neg = [-x for x in coef_pool if x > 0]
        coef_pool_all = sorted(set(coef_pool + coef_pool_neg + [0]))
        # Remove 0 since coefficients must be nonzero
        coef_pool_all = [c for c in coef_pool_all if c != 0]

        for restart in range(15):
            if time.time() > deadline - 3:
                break

            cur_exp = list(best_exp)
            cur_coef = list(best_coef)
            cur_nz = square_nonzero(cur_exp, cur_coef)

            # Intensive coefficient-only local search
            for it in range(8000):
                if time.time() > deadline - 3:
                    break

                exp_copy = list(cur_exp)
                coef_copy = list(cur_coef)

                r = random.random()
                if r < 0.50:
                    # Single coefficient change - most common, focus on fine-tuning
                    idx = random.randint(0, 11)
                    new_c = random.choice(coef_pool_all)
                    if new_c != coef_copy[idx]:
                        coef_copy[idx] = new_c
                elif r < 0.70:
                    # Double coefficient change
                    i, j = random.sample(range(12), 2)
                    coef_copy[i] = random.choice(coef_pool_all)
                    coef_copy[j] = random.choice(coef_pool_all)
                elif r < 0.80:
                    # Triple coefficient change
                    idxs = random.sample(range(12), 3)
                    for idx in idxs:
                        coef_copy[idx] = random.choice(coef_pool_all)
                elif r < 0.90:
                    # Exponent shift
                    idx = random.randint(0, 11)
                    shift = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                    new_e = exp_copy[idx] + shift
                    if 0 <= new_e < 41 and new_e not in exp_copy:
                        exp_copy[idx] = new_e
                        exp_copy.sort()
                    else:
                        continue
                else:
                    # Two exponent shifts
                    i, j = random.sample(range(12), 2)
                    si = random.choice([-3, -2, -1, 1, 2, 3])
                    sj = random.choice([-3, -2, -1, 1, 2, 3])
                    new_ei = exp_copy[i] + si
                    new_ej = exp_copy[j] + sj
                    if (0 <= new_ei < 41 and 0 <= new_ej < 41 and
                            new_ei not in exp_copy and new_ej not in exp_copy and
                            new_ei != new_ej):
                        exp_copy[i] = new_ei
                        exp_copy[j] = new_ej
                        exp_copy.sort()
                    else:
                        continue

                if len(set(exp_copy)) != 12:
                    continue
                if any(e < 0 or e >= 41 for e in exp_copy):
                    continue
                if any(c == 0 or abs(c) > 64 for c in coef_copy):
                    continue

                nz = square_nonzero(exp_copy, coef_copy)
                if nz < cur_nz:
                    cur_nz = nz
                    cur_exp = list(exp_copy)
                    cur_coef = list(coef_copy)

                    if nz < best_nz:
                        best_nz = nz
                        max_deg = max(cur_exp)
                        poly_new = [0] * (max_deg + 1)
                        for e, c in zip(cur_exp, cur_coef):
                            poly_new[e] = c
                        best_poly = poly_new
                        best_exp = list(cur_exp)
                        best_coef = list(cur_coef)

        # Final polish: exhaustive coefficient neighborhood around best
        if time.time() < deadline - 2:
            cur_exp = list(best_exp)
            cur_coef = list(best_coef)
            cur_nz = square_nonzero(cur_exp, cur_coef)
            
            # Try all single coefficient changes
            for idx in range(12):
                if time.time() > deadline - 2:
                    break
                for new_c in coef_pool_all:
                    if new_c == cur_coef[idx] or abs(new_c) > 64:
                        continue
                    test_coef = list(cur_coef)
                    test_coef[idx] = new_c
                    nz = square_nonzero(cur_exp, test_coef)
                    if nz < cur_nz:
                        cur_nz = nz
                        cur_coef = test_coef
            
            # Try all double coefficient changes (sampled)
            for _ in range(500):
                if time.time() > deadline - 2:
                    break
                i, j = random.sample(range(12), 2)
                ci = random.choice(coef_pool_all)
                cj = random.choice(coef_pool_all)
                if ci == cur_coef[i] and cj == cur_coef[j]:
                    continue
                test_coef = list(cur_coef)
                test_coef[i] = ci
                test_coef[j] = cj
                if any(abs(c) > 64 or c == 0 for c in test_coef):
                    continue
                nz = square_nonzero(cur_exp, test_coef)
                if nz < cur_nz:
                    cur_nz = nz
                    cur_coef = test_coef

            if cur_nz < best_nz:
                best_nz = cur_nz
                max_deg = max(cur_exp)
                poly_new = [0] * (max_deg + 1)
                for e, c in zip(cur_exp, cur_coef):
                    poly_new[e] = c
                best_poly = poly_new
                best_exp = list(cur_exp)
                best_coef = list(cur_coef)

    # Final fallback
    if best_poly is None:
        best_poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    if not validate(best_poly):
        best_poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    return best_poly

# EVOLVE-BLOCK-END
