import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 78  # 78s guard, 12s margin

    best_poly = [1] * 12
    best_count = 23

    def count_nonzero_square_sparse(nz):
        """nz: list of (exponent, coeff). Returns # nonzero terms in exact square."""
        sq = {}
        n = len(nz)
        for i in range(n):
            ei, ci = nz[i]
            for j in range(i, n):
                ej, cj = nz[j]
                deg = ei + ej
                val = ci * cj if i == j else 2 * ci * cj
                sq[deg] = sq.get(deg, 0) + val
        return sum(1 for v in sq.values() if v != 0)

    # Strategy 1: Arithmetic progression with sign patterns
    for d in range(1, 5):
        for a in range(0, 5):
            max_deg = a + 11 * d
            if max_deg >= 41:
                continue
            exponents = [a + i * d for i in range(12)]
            for signs in itertools.product([-1, 1], repeat=12):
                if time.time() > deadline:
                    return best_poly
                nz = [(exponents[i], signs[i]) for i in range(12)]
                cnt = count_nonzero_square_sparse(nz)
                if cnt < best_count:
                    best_count = cnt
                    coeffs = [0] * (max_deg + 1)
                    for i in range(12):
                        coeffs[exponents[i]] = signs[i]
                    best_poly = coeffs

    # Strategy 2: Two-cluster (same 6-term shape, shifted)
    for k in range(1, 8):
        for m in range(k + 1, k + 15):
            max_deg = m + k
            if max_deg >= 41:
                break
            for ex1 in itertools.combinations(range(k + 1), 6):
                if time.time() > deadline:
                    return best_poly
                ex2 = tuple(e + m for e in ex1)
                for signs1 in itertools.product([-1, 1], repeat=6):
                    for mode in [0, 1]:
                        signs2 = signs1 if mode == 0 else tuple(-s for s in signs1)
                        nz = [(ex1[i], signs1[i]) for i in range(6)] + [(ex2[i], signs2[i]) for i in range(6)]
                        cnt = count_nonzero_square_sparse(nz)
                        if cnt < best_count:
                            best_count = cnt
                            coeffs = [0] * (max_deg + 1)
                            for i in range(6):
                                coeffs[ex1[i]] = signs1[i]
                                coeffs[ex2[i]] = signs2[i]
                            best_poly = coeffs

    # Strategy 3: Q(x) ± x^m Q(x) with 6-term Q
    for m in range(2, 25):
        if 2 * m > 40:
            break
        for q_len in range(6, min(m, 41 - m) + 1):
            for ex in itertools.combinations(range(q_len), 6):
                if time.time() > deadline:
                    return best_poly
                for signs in itertools.product([-1, 1], repeat=6):
                    for mode in [0, 1]:
                        nz = [(ex[i], signs[i]) for i in range(6)]
                        for i in range(6):
                            c = signs[i] if mode == 0 else -signs[i]
                            nz.append((m + ex[i], c))
                        cnt = count_nonzero_square_sparse(nz)
                        if cnt < best_count:
                            best_count = cnt
                            coeffs = [0] * (m + q_len)
                            for i in range(6):
                                coeffs[ex[i]] = signs[i]
                                coeffs[m + ex[i]] = signs[i] if mode == 0 else -signs[i]
                            best_poly = coeffs

    # Strategy 4: Randomized search
    rng = random.Random(42)
    for _ in range(500000):
        if time.time() > deadline:
            return best_poly
        exponents = sorted(rng.sample(range(41), 12))
        nz = [(e, rng.choice([-2, -1, 1, 2])) for e in exponents]
        cnt = count_nonzero_square_sparse(nz)
        if cnt < best_count:
            best_count = cnt
            coeffs = [0] * 41
            for e, c in nz:
                coeffs[e] = c
            best_poly = coeffs

    return best_poly
