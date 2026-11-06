"""
Sparse squares of 12-term integer polynomials.
Strategy: structured candidate generation (APs, two-AP unions, symmetric
exponent layouts with diverse coefficient patterns) followed by simulated
annealing refinement (which escapes local minima of the rugged
nonzero-terms landscape better than greedy hill climbing).
"""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _square_count_from_exp(exp, coeff):
    """Nonzero-term count of the exact square of a 12-term poly given as
    parallel lists of exponents and (nonzero) coefficients. Exact via bincount."""
    e = np.asarray(exp, dtype=np.int64)
    c = np.asarray(coeff, dtype=np.int64)
    degs = e[:, None] + e[None, :]
    vals = c[:, None] * c[None, :]
    result = np.bincount(degs.ravel(), weights=vals.ravel(), minlength=82)
    return int(np.count_nonzero(result))

def _poly_from_exp(exp, coeff):
    max_deg = max(exp)
    poly = [0] * (max_deg + 1)
    for e, c in zip(exp, coeff):
        poly[e] = c
    return poly

def _validate(poly):
    if not poly or len(poly) > 41:
        return False
    if poly[-1] == 0:
        return False
    if sum(1 for x in poly if x != 0) != 12:
        return False
    if any(abs(x) > 64 for x in poly):
        return False
    return True

def _extract_exp_coeff(poly):
    exp = [i for i, v in enumerate(poly) if v != 0]
    coeff = [poly[i] for i in exp]
    return exp, coeff

def _square_count(poly):
    exp, coeff = _extract_exp_coeff(poly)
    if len(exp) != 12:
        return 10**9
    return _square_count_from_exp(exp, coeff)

def _simulated_annealing(exp0, coeff0, deadline, iters_budget):
    """SA over the 12 (exponent, coefficient) pairs. Moves: change a
    coefficient value, shift an exponent, or swap two terms. Accepts worse
    moves with a cooling schedule to escape local minima. Returns best
    (exp, coeff, score)."""
    exp = exp0[:]
    coeff = coeff0[:]
    cur_score = _square_count_from_exp(exp, coeff)
    best_exp = exp[:]
    best_coeff = coeff[:]
    best_score = cur_score

    temp = 4.0
    cool = 0.9990
    n = len(exp)
    for _ in range(iters_budget):
        if time.time() > deadline:
            break
        m = random.random()
        if m < 0.45:
            # change coefficient value
            i = random.randrange(n)
            old = coeff[i]
            new = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
            if new == old:
                continue
            coeff[i] = new
            new_score = _square_count_from_exp(exp, coeff)
            delta = new_score - cur_score
            if delta <= 0 or random.random() < np.exp(-delta / temp):
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_exp = exp[:]
                    best_coeff = coeff[:]
            else:
                coeff[i] = old
        elif m < 0.80:
            # shift exponent
            i = random.randrange(n)
            shift = random.choice([-1, -2, 1, 2])
            new_e = exp[i] + shift
            if new_e < 0 or new_e >= 41:
                continue
            if new_e in exp:
                continue
            exp[i] = new_e
            new_score = _square_count_from_exp(exp, coeff)
            delta = new_score - cur_score
            if delta <= 0 or random.random() < np.exp(-delta / temp):
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_exp = exp[:]
                    best_coeff = coeff[:]
            else:
                exp[i] = exp[i] - shift
        else:
            # swap two terms (exponents + coefficients)
            if n < 2:
                continue
            i, j = random.sample(range(n), 2)
            exp[i], exp[j] = exp[j], exp[i]
            coeff[i], coeff[j] = coeff[j], coeff[i]
            new_score = _square_count_from_exp(exp, coeff)
            delta = new_score - cur_score
            if delta <= 0 or random.random() < np.exp(-delta / temp):
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_exp = exp[:]
                    best_coeff = coeff[:]
            else:
                exp[i], exp[j] = exp[j], exp[i]
                coeff[i], coeff[j] = coeff[j], coeff[i]
        temp *= cool
    return best_exp, best_coeff, best_score

def _build_coeff_patterns():
    pats = []
    for period in [1, 2, 3, 4, 6, 12]:
        for phase in range(period):
            pats.append([1 if ((i + phase) % (2 * period)) < period else -1 for i in range(12)])
    pats += [
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
        [1, 2, -1, -2, 1, 2, -1, -2, 1, 2, -1, -2],
        [1, -2, 2, -1, 1, -2, 2, -1, 1, -2, 2, -1],
        [2, -1, 1, -2, 2, -1, 1, -2, 2, -1, 1, -2],
        [1, 1, -2, -1, 1, 1, -2, -1, 1, 1, -2, -1],
        [3, 1, -1, -3, -1, 1, 3, 1, -1, -3, -1, 1],
        [1, 3, -1, -3, 1, 3, -1, -3, 1, 3, -1, -3],
        [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
        [2, 2, -2, -2, 2, 2, -2, -2, 2, 2, -2, -2],
        [1, 2, 2, -1, -2, -2, 1, 2, 2, -1, -2, -2],
        [1, -1, 2, -2, 1, -1, 2, -2, 1, -1, 2, -2],
    ]
    return pats

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill

    best_score = 10**9
    best_exp = None
    best_coeff = None

    def try_candidate(exp, coeff):
        nonlocal best_score, best_exp, best_coeff
        if len(exp) != 12 or len(set(exp)) != 12:
            return
        if any(e < 0 or e >= 41 for e in exp):
            return
        if any(c == 0 or abs(c) > 64 for c in coeff):
            return
        s = _square_count_from_exp(exp, coeff)
        if s < best_score:
            best_score = s
            best_exp = exp[:]
            best_coeff = coeff[:]

    pats = _build_coeff_patterns()

    # Phase 1: single arithmetic progressions with many coefficient patterns
    for d in range(1, 8):
        if time.time() > deadline - 35:
            break
        max_a = min(40 - 11 * d, 15)
        for a in range(0, max_a + 1):
            if time.time() > deadline - 35:
                break
            exp = [a + i * d for i in range(12)]
            if exp[-1] >= 41:
                continue
            for cp in pats:
                if time.time() > deadline - 35:
                    break
                try_candidate(exp, cp)

    # Phase 2: two-AP unions (6+6 terms)
    if time.time() < deadline - 30:
        for d1 in range(1, 5):
            for d2 in range(1, 5):
                if time.time() > deadline - 25:
                    break
                for a1 in range(0, 12):
                    for gap in range(3, 22):
                        if time.time() > deadline - 30:
                            break
                        a2 = a1 + gap
                        e1 = [a1 + i * d1 for i in range(6)]
                        e2 = [a2 + i * d2 for i in range(6)]
                        exp = sorted(e1 + e2)
                        if len(set(exp)) != 12 or any(e >= 41 for e in exp):
                            continue
                        for s1 in [1, -1]:
                            for s2 in [1, -1]:
                                if time.time() > deadline - 30:
                                    break
                                try_candidate(exp, [s1] * 6 + [s2] * 6)
                                c_alt = [s1 * (1 if i % 2 == 0 else -1) for i in range(6)] + \
                                        [s2 * (1 if i % 2 == 0 else -1) for i in range(6)]
                                try_candidate(exp, c_alt)
                                c_blk = [s1 * (1 if (i // 2) % 2 == 0 else -1) for i in range(6)] + \
                                        [s2 * (1 if (i // 2) % 2 == 0 else -1) for i in range(6)]
                                try_candidate(exp, c_blk)

    # Phase 3: symmetric exponent layouts
    if time.time() < deadline - 20:
        for center in range(6, 35):
            if time.time() > deadline - 20:
                break
            for offsets in [
                [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 7], [1, 2, 3, 4, 6, 8],
                [1, 2, 3, 5, 7, 9], [1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12],
                [1, 2, 4, 5, 7, 8], [1, 3, 4, 6, 7, 9], [1, 2, 4, 6, 8, 10],
                [2, 3, 4, 5, 6, 7],
            ]:
                if time.time() > deadline - 20:
                    break
                exp = sorted([center - o for o in offsets] + [center + o for o in offsets])
                if any(e < 0 or e >= 41 for e in exp) or len(set(exp)) != 12:
                    continue
                for cp in pats[:20]:
                    if time.time() > deadline - 20:
                        break
                    try_candidate(exp, cp)

    # Phase 4: randomized structured candidates
    random.seed(42)
    if time.time() < deadline - 18:
        for _ in range(60000):
            if time.time() > deadline - 18:
                break
            r = random.random()
            if r < 0.20:
                d = random.randint(1, 6)
                max_a = max(0, 40 - 11 * d)
                a = random.randint(0, max_a)
                exp = [a + i * d for i in range(12)]
                if exp[-1] >= 41:
                    continue
            elif r < 0.40:
                d1 = random.randint(1, 4)
                d2 = random.randint(1, 4)
                a1 = random.randint(0, 10)
                gap = random.randint(4, 18)
                a2 = a1 + gap
                exp = [a1 + i * d1 for i in range(6)] + [a2 + i * d2 for i in range(6)]
                if any(e >= 41 for e in exp) or len(set(exp)) != 12:
                    continue
                exp.sort()
            elif r < 0.60:
                center = random.randint(6, 34)
                offsets = sorted(random.sample(range(1, 20), 6))
                exp = [center - o for o in offsets] + [center + o for o in offsets]
                if any(e < 0 or e >= 41 for e in exp) or len(set(exp)) != 12:
                    continue
                exp.sort()
            elif r < 0.80:
                window = random.randint(14, 28)
                base = random.randint(0, max(0, 40 - window))
                exp = sorted(random.sample(range(base, min(base + window, 41)), 12))
            else:
                d1 = random.randint(1, 3)
                d2 = random.randint(1, 3)
                d3 = random.randint(1, 3)
                a1 = random.randint(0, 5)
                g1 = random.randint(3, 8)
                a2 = a1 + g1
                g2 = random.randint(3, 8)
                a3 = a2 + g2
                exp = [a1 + i * d1 for i in range(4)] + [a2 + i * d2 for i in range(4)] + \
                      [a3 + i * d3 for i in range(4)]
                if any(e >= 41 for e in exp) or len(set(exp)) != 12:
                    continue
                exp.sort()
            # random coefficient pattern with small magnitudes
            mag = random.choice([1, 1, 1, 2, 2, 3])
            coeff = [random.choice([-1, 1]) * mag for _ in range(12)]
            try_candidate(exp, coeff)

    # Phase 5: simulated annealing refinement of the best structured candidate
    if best_exp is not None and time.time() < deadline - 5:
        # Refine the current best a few times with different seeds / temperatures
        sa_deadline = deadline - 5
        for trial in range(3):
            if time.time() > sa_deadline:
                break
            random.seed(1000 + trial)
            e, c, s = _simulated_annealing(best_exp, best_coeff, sa_deadline, iters_budget=400000)
            if s < best_score:
                best_score = s
                best_exp = e
                best_coeff = c

    # Build final polynomial list
    if best_exp is None:
        # fallback: simple alternating AP
        poly = [0] * 12
        for i in range(12):
            poly[i] = 1 if i % 2 == 0 else -1
        return poly

    poly = _poly_from_exp(best_exp, best_coeff)
    if not _validate(poly):
        poly = [0] * 12
        for i in range(12):
            poly[i] = 1 if i % 2 == 0 else -1
    return poly

# EVOLVE-BLOCK-END
