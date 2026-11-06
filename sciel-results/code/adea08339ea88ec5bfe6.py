"""Sparse squares of 12-term integer polynomials – structural + SA hybrid."""
import time
import random
import numpy as np

def construct():
    start = time.time()
    deadline = start + 85.0

    def square_nonzero_exact(exponents, coeffs):
        c = np.asarray(coeffs, dtype=np.int64)
        e = np.asarray(exponents, dtype=np.int64)
        degs = e[:, None] + e[None, :]
        vals = c[:, None] * c[None, :]
        result = np.bincount(degs.ravel(), weights=vals.ravel(), minlength=82)
        return int(np.count_nonzero(result))

    def is_valid(exp, coef):
        if len(exp) != 12 or len(coef) != 12:
            return False
        if len(set(exp)) != 12:
            return False
        if any(e < 0 or e >= 41 for e in exp):
            return False
        if any(c == 0 or abs(c) > 64 for c in coef):
            return False
        return True

    def make_poly(exp, coef):
        max_deg = max(exp)
        poly = [0] * (max_deg + 1)
        for e, c in zip(exp, coef):
            poly[e] = int(c)
        return poly

    # Core coefficient patterns for structured cancellation
    core_patterns = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
        [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
        [1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1],
        [1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1],
        [1, 2, -1, -2, 1, 2, -1, -2, 1, 2, -1, -2],
        [1, -2, 2, -1, 1, -2, 2, -1, 1, -2, 2, -1],
        [2, -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2],
        [1, 1, -2, -1, 1, 1, -2, -1, 1, 1, -2, -1],
        [3, 1, -1, -3, -1, 1, 3, 1, -1, -3, -1, 1],
        [1, 3, -1, -3, 1, 3, -1, -3, 1, 3, -1, -3],
        [1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1],
        [1, 2, -2, -1, 1, 2, -2, -1, 1, 2, -2, -1],
        [1, -2, -2, 1, 1, -2, -2, 1, 1, -2, -2, 1],
        [2, 1, -1, -2, -1, 1, 2, 1, -1, -2, -1, 1],
        [1, 1, 2, -1, -2, -1, 1, 1, 2, -1, -2, -1],
        [1, -1, 2, -2, 1, -1, 2, -2, 1, -1, 2, -2],
        [2, -2, 1, -1, 2, -2, 1, -1, 2, -2, 1, -1],
        [1, 3, -3, -1, 1, 3, -3, -1, 1, 3, -3, -1],
    ]

    # Generate periodic sign patterns
    for period in [1, 2, 3, 4, 6, 12]:
        for phase in range(period):
            pat = [1 if ((i + phase) % (2 * period)) < period else -1 for i in range(12)]
            if pat not in core_patterns:
                core_patterns.append(pat)

    best_exp = None
    best_coef = None
    best_nz = float('inf')

    def try_candidate(exp, coef):
        nonlocal best_exp, best_coef, best_nz
        if not is_valid(exp, coef):
            return
        nz = square_nonzero_exact(exp, coef)
        if nz < best_nz:
            best_nz = nz
            best_exp = list(exp)
            best_coef = list(coef)

    # Phase 1: Structured search over AP exponents with core patterns
    for d in range(1, 8):
        if time.time() > deadline - 40:
            break
        max_a = min(40 - 11 * d, 15)
        for a in range(0, max_a + 1):
            if time.time() > deadline - 40:
                break
            exp = [a + i * d for i in range(12)]
            for cp in core_patterns:
                try_candidate(exp, cp)

    # Phase 2: Two-AP union structures (6+6 terms)
    if time.time() < deadline - 30:
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
            [1, 3, -3, -1, 3, -3],
            [1, 1, -1, -1, -1, -1],
        ]
        all_pats_6 = sign_pats_6 + extra6

        for d1 in range(1, 5):
            for d2 in range(1, 5):
                if time.time() > deadline - 30:
                    break
                for a1 in range(0, 14):
                    for gap in range(3, 24):
                        if time.time() > deadline - 30:
                            break
                        a2 = a1 + gap
                        exp1 = [a1 + i * d1 for i in range(6)]
                        exp2 = [a2 + i * d2 for i in range(6)]
                        exp = sorted(exp1 + exp2)
                        if len(set(exp)) != 12 or any(e >= 41 for e in exp):
                            continue
                        for pq in all_pats_6:
                            for pr in all_pats_6:
                                if time.time() > deadline - 30:
                                    break
                                coef = list(pq) + list(pr)
                                try_candidate(exp, coef)

    # Phase 3: Symmetric structures
    if time.time() < deadline - 20:
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
            [1, 2, 3, 4, 5, 8],
            [1, 2, 3, 5, 6, 9],
        ]
        for center in range(6, 35):
            if time.time() > deadline - 20:
                break
            for offsets in offset_sets:
                if time.time() > deadline - 20:
                    break
                exp = sorted([center - o for o in offsets] + [center + o for o in offsets])
                if any(e < 0 or e >= 41 for e in exp) or len(set(exp)) != 12:
                    continue
                for cp in core_patterns[:20]:
                    try_candidate(exp, cp)

    # Phase 4: Simulated annealing from best and diverse starts
    def simulated_annealing(initial_exp, initial_coef, max_iters, t_start, t_end, dl):
        cur_exp = list(initial_exp)
        cur_coef = list(initial_coef)
        cur_nz = square_nonzero_exact(cur_exp, cur_coef)

        best_e = list(cur_exp)
        best_c = list(cur_coef)
        best_n = cur_nz

        n = 12
        for it in range(max_iters):
            if time.time() > dl:
                break

            progress = it / max(max_iters - 1, 1)
            temp = t_start * (t_end / t_start) ** progress

            r = random.random()
            new_exp = list(cur_exp)
            new_coef = list(cur_coef)

            if r < 0.30:
                idx = random.randint(0, n - 1)
                new_c = random.choice([-8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8])
                if new_c == cur_coef[idx]:
                    continue
                new_coef[idx] = new_c
            elif r < 0.50:
                i, j = random.sample(range(n), 2)
                new_coef[i] = random.choice([-6, -4, -3, -2, -1, 1, 2, 3, 4, 6])
                new_coef[j] = random.choice([-6, -4, -3, -2, -1, 1, 2, 3, 4, 6])
            elif r < 0.65:
                idx = random.randint(0, n - 1)
                shift = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                new_e = cur_exp[idx] + shift
                if 0 <= new_e < 41 and new_e not in cur_exp:
                    new_exp[idx] = new_e
                    new_exp.sort()
                else:
                    continue
            elif r < 0.78:
                i, j = random.sample(range(n), 2)
                si = random.choice([-3, -2, -1, 1, 2, 3])
                sj = random.choice([-3, -2, -1, 1, 2, 3])
                new_ei = cur_exp[i] + si
                new_ej = cur_exp[j] + sj
                if (0 <= new_ei < 41 and 0 <= new_ej < 41 and
                        new_ei not in cur_exp and new_ej not in cur_exp and
                        new_ei != new_ej):
                    new_exp[i] = new_ei
                    new_exp[j] = new_ej
                    new_exp.sort()
                else:
                    continue
            elif r < 0.88:
                i, j = random.sample(range(n), 2)
                new_exp[i], new_exp[j] = new_exp[j], new_exp[i]
                new_exp.sort()
            else:
                idx = random.randint(0, n - 1)
                new_exp[idx] = random.randint(0, 40)
                new_exp.sort()
                if len(set(new_exp)) != n:
                    continue
                for _ in range(random.randint(1, 4)):
                    ci = random.randint(0, n - 1)
                    new_coef[ci] = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])

            if not is_valid(new_exp, new_coef):
                continue

            new_nz = square_nonzero_exact(new_exp, new_coef)

            delta = new_nz - cur_nz
            if delta <= 0:
                cur_exp = new_exp
                cur_coef = new_coef
                cur_nz = new_nz
            else:
                if temp > 0 and random.random() < np.exp(-delta / max(temp, 0.01)):
                    cur_exp = new_exp
                    cur_coef = new_coef
                    cur_nz = new_nz

            if cur_nz < best_n:
                best_n = cur_nz
                best_e = list(cur_exp)
                best_c = list(cur_coef)

        return best_e, best_c, best_n

    if best_exp is not None and time.time() < deadline - 5:
        e, c, nz = simulated_annealing(best_exp, best_coef, 15000, 5.0, 0.01, deadline - 3)
        if nz < best_nz:
            best_nz = nz
            best_exp = e
            best_coef = c

    # SA from diverse initial points
    random.seed(99)
    diverse_starts = []
    for _ in range(8):
        r = random.random()
        if r < 0.3:
            d = random.randint(1, 5)
            a = random.randint(0, max(0, 40 - 11 * d))
            exp = [a + i * d for i in range(12)]
        elif r < 0.6:
            center = random.randint(8, 32)
            offsets = sorted(random.sample(range(1, 18), 6))
            exp = sorted([center - o for o in offsets] + [center + o for o in offsets])
        else:
            exp = sorted(random.sample(range(41), 12))

        if is_valid(exp, [1] * 12):
            coef = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            diverse_starts.append((exp, coef))

    for exp, coef in diverse_starts:
        if time.time() > deadline - 3:
            break
        e, c, nz = simulated_annealing(exp, coef, 8000, 4.0, 0.01, deadline - 3)
        if nz < best_nz:
            best_nz = nz
            best_exp = e
            best_coef = c

    # Phase 5: Focused local refinement
    if best_exp is not None and time.time() < deadline - 2:
        cur_exp = list(best_exp)
        cur_coef = list(best_coef)
        cur_nz = best_nz

        for it in range(20000):
            if time.time() > deadline - 1:
                break

            new_exp = list(cur_exp)
            new_coef = list(cur_coef)
            r = random.random()

            if r < 0.4:
                idx = random.randint(0, 11)
                new_c = random.choice([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
                if new_c != cur_coef[idx]:
                    new_coef[idx] = new_c
            elif r < 0.7:
                idx = random.randint(0, 11)
                shift = random.choice([-3, -2, -1, 1, 2, 3])
                new_e = cur_exp[idx] + shift
                if 0 <= new_e < 41 and new_e not in cur_exp:
                    new_exp[idx] = new_e
                    new_exp.sort()
                else:
                    continue
            else:
                i, j = random.sample(range(12), 2)
                si = random.choice([-2, -1, 1, 2])
                sj = random.choice([-2, -1, 1, 2])
                new_ei = cur_exp[i] + si
                new_ej = cur_exp[j] + sj
                if (0 <= new_ei < 41 and 0 <= new_ej < 41 and
                        new_ei not in cur_exp and new_ej not in cur_exp and
                        new_ei != new_ej):
                    new_exp[i] = new_ei
                    new_exp[j] = new_ej
                    new_exp.sort()
                else:
                    continue

            if not is_valid(new_exp, new_coef):
                continue

            new_nz = square_nonzero_exact(new_exp, new_coef)
            if new_nz < cur_nz:
                cur_nz = new_nz
                cur_exp = new_exp
                cur_coef = new_coef
                if cur_nz < best_nz:
                    best_nz = cur_nz
                    best_exp = list(cur_exp)
                    best_coef = list(cur_coef)

    # Final validation
    def validate(poly):
        if not poly or len(poly) > 41:
            return False
        if poly[-1] == 0:
            return False
        for c in poly:
            if not isinstance(c, int) or c < -64 or c > 64:
                return False
        if sum(1 for c in poly if c != 0) != 12:
            return False
        return True

    if best_exp is None or not is_valid(best_exp, best_coef):
        best_exp = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 39, 40]
        best_coef = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    poly = make_poly(best_exp, best_coef)

    if not validate(poly):
        poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    return poly
