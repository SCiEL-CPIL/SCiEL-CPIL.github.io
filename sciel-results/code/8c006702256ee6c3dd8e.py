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
        """Fast exact square nonzero count via dict."""
        result = {}
        n = len(exponents)
        for i in range(n):
            ei = exponents[i]
            ci = coeffs[i]
            for j in range(i, n):
                d = ei + exponents[j]
                v = ci * coeffs[j]
                if i != j:
                    v *= 2
                result[d] = result.get(d, 0) + v
        return sum(1 for v in result.values() if v != 0)

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
            if cstyle < 0.25:
                coeffs = [random.choice([-1, 1]) for _ in range(12)]
            elif cstyle < 0.45:
                coeffs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            elif cstyle < 0.65:
                pat = random.choice(all_patterns)
                coeffs = list(pat)
            elif cstyle < 0.80:
                coeffs = [random.randint(-3, 3) for _ in range(12)]
                for i in range(12):
                    while coeffs[i] == 0:
                        coeffs[i] = random.randint(1, 3) * random.choice([-1, 1])
            else:
                coeffs = [random.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for _ in range(12)]

            if all(abs(c) <= 64 and c != 0 for c in coeffs):
                try_candidate(exponents, coeffs)

    random_search(150000)

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

    # ---------- Phase 6: Enhanced local search with diverse seeds ----------
    if best_poly is not None and time.time() < deadline - 5:
        best_exp = [i for i, c in enumerate(best_poly) if c != 0]
        best_coef = [c for c in best_poly if c != 0]

        # Build diverse seed set for local search
        seeds = [(list(best_exp), list(best_coef))]

        # Add structured seeds
        for d in [1, 2, 3]:
            a = random.randint(0, max(0, 40 - 11 * d))
            se = [a + i * d for i in range(12)]
            if se[-1] < 41:
                sc = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
                seeds.append((se, sc))

        # Two-AP seeds
        for _ in range(5):
            d1 = random.randint(1, 4)
            d2 = random.randint(1, 4)
            a1 = random.randint(0, 10)
            gap = random.randint(4, 15)
            a2 = a1 + gap
            se = sorted([a1 + i * d1 for i in range(6)] + [a2 + i * d2 for i in range(6)])
            if len(set(se)) == 12 and se[-1] < 41:
                sc = [random.choice([-1, 1]) for _ in range(12)]
                seeds.append((se, sc))

        # Symmetric seeds
        for _ in range(5):
            center = random.randint(8, 30)
            offs = sorted(random.sample(range(1, 16), 6))
            se = sorted([center - o for o in offs] + [center + o for o in offs])
            if all(0 <= e < 41 for e in se) and len(set(se)) == 12:
                sc = [random.choice([-1, 1]) for _ in range(12)]
                seeds.append((se, sc))

        # Product-of-6 seeds
        for _ in range(5):
            d1 = random.randint(1, 4)
            d2 = random.randint(1, 4)
            a1 = random.randint(0, 12)
            a2 = random.randint(0, 12)
            qe = [a1 + i * d1 for i in range(6)]
            re = [a2 + i * d2 for i in range(6)]
            if len(set(qe + re)) == 12 and all(e < 41 for e in qe + re):
                se = sorted(qe + re)
                qm = dict(zip(qe, [random.choice([-1, 1]) for _ in range(6)]))
                rm = dict(zip(re, [random.choice([-1, 1]) for _ in range(6)]))
                sc = [qm.get(e, rm.get(e, 1)) for e in se]
                seeds.append((se, sc))

        # Run local search from each seed
        for seed_idx, (s_exp, s_coef) in enumerate(seeds):
            if time.time() > deadline - 3:
                break

            cur_exp = list(s_exp)
            cur_coef = list(s_coef)
            cur_nz = square_nonzero(cur_exp, cur_coef)

            # Coordinate descent: try all single moves, take best improvement
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                best_move_nz = cur_nz
                best_move_exp = None
                best_move_coef = None

                # Try coefficient changes
                for idx in range(12):
                    for nc in [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 12, -8, -12]:
                        if nc == cur_coef[idx] or abs(nc) > 64 or nc == 0:
                            continue
                        tc = list(cur_coef)
                        tc[idx] = nc
                        nz = square_nonzero(cur_exp, tc)
                        if nz < best_move_nz:
                            best_move_nz = nz
                            best_move_exp = list(cur_exp)
                            best_move_coef = tc

                # Try exponent shifts
                for idx in range(12):
                    for shift in [-4, -3, -2, -1, 1, 2, 3, 4]:
                        ne = cur_exp[idx] + shift
                        if 0 <= ne < 41 and ne not in cur_exp:
                            te = list(cur_exp)
                            te[idx] = ne
                            te.sort()
                            nz = square_nonzero(te, cur_coef)
                            if nz < best_move_nz:
                                best_move_nz = nz
                                best_move_exp = te
                                best_move_coef = list(cur_coef)

                # Try paired exponent shifts
                for i in range(12):
                    for j in range(i + 1, 12):
                        for si in [-2, -1, 1, 2]:
                            for sj in [-2, -1, 1, 2]:
                                ne_i = cur_exp[i] + si
                                ne_j = cur_exp[j] + sj
                                if (0 <= ne_i < 41 and 0 <= ne_j < 41 and
                                        ne_i not in cur_exp and ne_j not in cur_exp and
                                        ne_i != ne_j):
                                    te = list(cur_exp)
                                    te[i] = ne_i
                                    te[j] = ne_j
                                    te.sort()
                                    nz = square_nonzero(te, cur_coef)
                                    if nz < best_move_nz:
                                        best_move_nz = nz
                                        best_move_exp = te
                                        best_move_coef = list(cur_coef)

                if best_move_exp is not None:
                    cur_exp = best_move_exp
                    cur_coef = best_move_coef
                    cur_nz = best_move_nz
                    improved = True

                    if cur_nz < best_nz:
                        best_nz = cur_nz
                        max_deg = max(cur_exp)
                        poly_new = [0] * (max_deg + 1)
                        for e, c in zip(cur_exp, cur_coef):
                            poly_new[e] = c
                        best_poly = poly_new
                        best_exp = list(cur_exp)
                        best_coef = list(cur_coef)

                # If no improvement via coordinate descent, do random perturbations
                if not improved:
                    for _ in range(500):
                        if time.time() > deadline - 3:
                            break
                        exp_copy = list(cur_exp)
                        coef_copy = list(cur_coef)
                        r = random.random()
                        if r < 0.4:
                            idx = random.randint(0, 11)
                            nc = random.choice([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 12])
                            if nc != coef_copy[idx]:
                                coef_copy[idx] = nc
                        else:
                            idx = random.randint(0, 11)
                            shift = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
                            ne = exp_copy[idx] + shift
                            if 0 <= ne < 41 and ne not in exp_copy:
                                exp_copy[idx] = ne
                                exp_copy.sort()
                            else:
                                continue
                        if len(set(exp_copy)) != 12 or any(e < 0 or e >= 41 for e in exp_copy):
                            continue
                        if any(c == 0 or abs(c) > 64 for c in coef_copy):
                            continue
                        nz = square_nonzero(exp_copy, coef_copy)
                        if nz < cur_nz:
                            cur_nz = nz
                            cur_exp = list(exp_copy)
                            cur_coef = list(coef_copy)
                            if cur_nz < best_nz:
                                best_nz = cur_nz
                                max_deg = max(cur_exp)
                                poly_new = [0] * (max_deg + 1)
                                for e, c in zip(cur_exp, cur_coef):
                                    poly_new[e] = c
                                best_poly = poly_new
                                best_exp = list(cur_exp)
                                best_coef = list(cur_coef)
                            break

    # Final fallback
    if best_poly is None:
        best_poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    if not validate(best_poly):
        best_poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    return best_poly

# EVOLVE-BLOCK-END
