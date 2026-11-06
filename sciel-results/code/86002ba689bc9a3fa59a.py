"""Rewritten program: structured generation + local descent on P^2 sparsity."""
import time
import itertools
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    def elapsed_ok():
        return time.time() < deadline

    def count_nonzero(coeffs):
        """Count nonzero terms after trimming trailing zeros."""
        i = len(coeffs)
        while i > 0 and coeffs[i - 1] == 0:
            i -= 1
        return i

    def square_nonzero(exponents, coefficients):
        """Return (nonzero_count, dict of exponent->coeff) for P^2."""
        sq = defaultdict(int)
        n = len(exponents)
        for i in range(n):
            ci = coefficients[i]
            ei = exponents[i]
            for j in range(i, n):
                e = ei + exponents[j]
                if i == j:
                    sq[e] += ci * ci
                else:
                    sq[e] += 2 * ci * coefficients[j]
        nz = sum(1 for v in sq.values() if v != 0)
        return nz, sq

    def make_poly(exponents, coefficients):
        max_deg = max(exponents) if exponents else 0
        coeffs = [0] * (max_deg + 1)
        for e, c in zip(exponents, coefficients):
            coeffs[e] += c
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs

    def is_valid(coeffs):
        if len(coeffs) > 41:
            return False
        if not coeffs or coeffs[-1] == 0:
            return False
        nz = sum(1 for c in coeffs if c != 0)
        if nz != 12:
            return False
        for c in coeffs:
            if c < -64 or c > 64:
                return False
        return True

    def nz_of_poly(coeffs):
        """Nonzero count of the square of the polynomial given as coeff list."""
        exponents = [i for i, c in enumerate(coeffs) if c != 0]
        coefficients = [c for c in coeffs if c != 0]
        nz, _ = square_nonzero(exponents, coefficients)
        return nz

    # State represented as (exponents tuple, coefficients tuple), 12 terms each.
    # We keep them as parallel lists.

    best_score = None
    best_exponents = None
    best_coefficients = None

    def consider(exponents, coefficients):
        nonlocal best_score, best_exponents, best_coefficients
        if len(exponents) != 12:
            return
        if len(set(exponents)) != 12:
            return
        if max(exponents) > 40:
            return
        for c in coefficients:
            if c == 0 or c < -64 or c > 64:
                return
        nz, _ = square_nonzero(exponents, coefficients)
        if best_score is None or nz < best_score:
            best_score = nz
            best_exponents = list(exponents)
            best_coefficients = list(coefficients)

    # Cleaner local search with better moves
    def local_search2(exponents, coefficients, max_iters=600):
        exponents = list(exponents)
        coefficients = list(coefficients)
        nz, _ = square_nonzero(exponents, coefficients)
        improved = True
        iters = 0
        while improved and iters < max_iters and elapsed_ok():
            improved = False
            iters += 1
            for i in range(12):
                if not elapsed_ok():
                    break
                orig_e = exponents[i]
                orig_c = coefficients[i]
                best_local_nz = nz
                best_local_e = orig_e
                best_local_c = orig_c
                # exponent candidates
                for d in [-3, -2, -1, 1, 2, 3, -5, 5, -8, 8, -12, 12, -1, 1]:
                    new_e = orig_e + d
                    if new_e < 0 or new_e > 40:
                        continue
                    others = exponents[:i] + exponents[i+1:]
                    if new_e in others:
                        continue
                    exponents[i] = new_e
                    nz2, _ = square_nonzero(exponents, coefficients)
                    if nz2 < best_local_nz:
                        best_local_nz = nz2
                        best_local_e = new_e
                    exponents[i] = orig_e
                # coefficient candidates
                for c_new in [-64, -32, -8, -4, -2, -1, 1, 2, 4, 8, 32, 64, -3, 3, -5, 5]:
                    if c_new == orig_c or c_new == 0:
                        continue
                    coefficients[i] = c_new
                    nz2, _ = square_nonzero(exponents, coefficients)
                    if nz2 < best_local_nz:
                        best_local_nz = nz2
                        best_local_c = c_new
                    coefficients[i] = orig_c
                if best_local_nz < nz:
                    exponents[i] = best_local_e
                    coefficients[i] = best_local_c
                    nz = best_local_nz
                    improved = True
        return exponents, coefficients, nz

    # --- Generate structured candidates ---
    candidates = []  # list of (exponents, coefficients)

    # 1. Sign patterns on consecutive exponents
    for mask in range(4096):
        signs = [1 if (mask >> i) & 1 else -1 for i in range(12)]
        exponents = list(range(12))
        candidates.append((exponents, signs))

    # 2. Products of binomials: P = prod_{k} (a_k + b_k x^{e_k})
    #    Expand to get terms; keep if exactly 12 nonzero and degree<=40.
    def expand_product(factors):
        # factors: list of (coef_list) where each is sparse dict exp->coef
        poly = {0: 1}
        for f in factors:
            new_poly = defaultdict(int)
            for e1, c1 in poly.items():
                for e2, c2 in f.items():
                    new_poly[e1 + e2] += c1 * c2
            poly = {e: c for e, c in new_poly.items() if c != 0}
            if len(poly) > 12:
                return None
        return poly

    # Try products of up to 12 binomials with small exponents
    binomial_exp_choices = [1, 2, 3, 4, 5, 6, 7, 8]
    for nbin in range(1, 13):
        if not elapsed_ok():
            break
        # choose nbin exponents (with replacement allowed but track)
        for exps in itertools.product(binomial_exp_choices, repeat=min(nbin, 6)):
            if not elapsed_ok():
                break
            if nbin > 6:
                continue  # limit combinations
            factors = []
            for e in exps:
                # binomial (1 + x^e) or (1 - x^e)
                for s in (1, -1):
                    pass
            # Just use (1 + x^e) for simplicity in this loop; sign variants below
            factors = [{0: 1, e: 1} for e in exps]
            poly = expand_product(factors)
            if poly is not None and len(poly) == 12:
                exponents = sorted(poly.keys())
                if max(exponents) <= 40:
                    coefficients = [poly[e] for e in exponents]
                    candidates.append((exponents, coefficients))
            # signed variant
            factors = [{0: 1, e: -1} for e in exps[:len(exps)//2]] + [{0: 1, e: 1} for e in exps[len(exps)//2:]]
            poly = expand_product(factors)
            if poly is not None and len(poly) == 12:
                exponents = sorted(poly.keys())
                if max(exponents) <= 40:
                    coefficients = [poly[e] for e in exponents]
                    candidates.append((exponents, coefficients))

    # 3. P = Q^2 where Q is a 6-term polynomial with structured exponents
    #    Geometric progression exponents for Q: 0, r, 2r, ..., 5r
    for r in range(1, 9):
        q_exps = [i * r for i in range(6)]
        if max(q_exps) * 2 > 40:
            continue
        # Q^2 exponents are sums q_e[i]+q_e[j]
        for qmask in range(64):
            if not elapsed_ok():
                break
            q_signs = [1 if (qmask >> i) & 1 else -1 for i in range(6)]
            # P = Q^2
            p_poly = defaultdict(int)
            for i in range(6):
                for j in range(i, 6):
                    e = q_exps[i] + q_exps[j]
                    if i == j:
                        p_poly[e] += q_signs[i] * q_signs[j]
                    else:
                        p_poly[e] += 2 * q_signs[i] * q_signs[j]
            p_poly = {e: c for e, c in p_poly.items() if c != 0}
            if len(p_poly) == 12:
                exponents = sorted(p_poly.keys())
                if max(exponents) <= 40:
                    coefficients = [p_poly[e] for e in exponents]
                    candidates.append((exponents, coefficients))

    # 4. P = (1 - x^k)^m * (something) to get 12 terms
    #    (1 - x^k)^m has m+1 terms. Combine two such.
    for k1 in range(1, 11):
        for m1 in range(1, 8):
            for k2 in range(1, 11):
                for m2 in range(1, 8):
                    if not elapsed_ok():
                        break
                    if (m1 + 1) + (m2 + 1) - 1 > 12:  # rough
                        continue
                    # (1 - x^k1)^m1 * (1 - x^k2)^m2
                    f1 = defaultdict(int)
                    for a in range(m1 + 1):
                        f1[a * k1] += ((-1) ** a) * 1  # binomial coeff = C(m1,a) but we want simple; use actual binomial
                    # Actually (1 - x^k)^m = sum C(m,a)(-1)^a x^{ka}
                    import math
                    f1 = {a * k1: math.comb(m1, a) * ((-1) ** a) for a in range(m1 + 1)}
                    f2 = {b * k2: math.comb(m2, b) * ((-1) ** b) for b in range(m2 + 1)}
                    poly = defaultdict(int)
                    for e1, c1 in f1.items():
                        for e2, c2 in f2.items():
                            poly[e1 + e2] += c1 * c2
                    poly = {e: c for e, c in poly.items() if c != 0}
                    if len(poly) == 12:
                        exponents = sorted(poly.keys())
                        if max(exponents) <= 40:
                            coefficients = [poly[e] for e in exponents]
                            if all(abs(c) <= 64 for c in coefficients):
                                candidates.append((exponents, coefficients))

    # 5. Random candidates
    rng = random.Random(12345)
    for _ in range(3000):
        if not elapsed_ok():
            break
        exponents = sorted(rng.sample(range(0, 41), 12))
        coefficients = [rng.choice([-2, -1, 1, 2]) for _ in range(12)]
        candidates.append((exponents, coefficients))

    # 6. New: Consecutive exponents with optimized coefficient search
    # Try all combinations of small coefficients for consecutive exponents
    # This is expensive, so we sample or use a heuristic.
    # We'll just add a few more random ones with consecutive exponents.
    for _ in range(1000):
        if not elapsed_ok():
            break
        exponents = list(range(12))
        coefficients = [rng.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
        candidates.append((exponents, coefficients))

    # --- Evaluate all candidates, keep best, then local search on top ones ---
    scored = []
    for exponents, coefficients in candidates:
        if not elapsed_ok():
            break
        if len(exponents) != 12 or len(set(exponents)) != 12:
            continue
        if max(exponents) > 40:
            continue
        if any(c == 0 or c < -64 or c > 64 for c in coefficients):
            continue
        nz, _ = square_nonzero(exponents, coefficients)
        scored.append((nz, exponents, coefficients))

    scored.sort(key=lambda x: x[0])

    # Local search on the best 200 candidates
    for nz, exponents, coefficients in scored[:200]:
        if not elapsed_ok():
            break
        if nz >= (best_score if best_score is not None else 10**9):
            # still might improve; but skip if already worse than current best by a lot
            if best_score is not None and nz > best_score + 5:
                continue
        new_e, new_c, new_nz = local_search2(exponents, coefficients, max_iters=200)
        consider(new_e, new_c)

    # Also local search from the overall best found so far, with more iterations
    if best_exponents is not None and elapsed_ok():
        new_e, new_c, new_nz = local_search2(best_exponents, best_coefficients, max_iters=2000)
        consider(new_e, new_c)

    # Final: also try a few more random restarts + local search if time
    if elapsed_ok():
        for _ in range(500):
            if not elapsed_ok():
                break
            exponents = sorted(rng.sample(range(0, 41), 12))
            coefficients = [rng.choice([-2, -1, 1, 2]) for _ in range(12)]
            new_e, new_c, new_nz = local_search2(exponents, coefficients, max_iters=100)
            consider(new_e, new_c)

    # Build final polynomial
    if best_exponents is None:
        return [1] * 12
    poly = make_poly(best_exponents, best_coefficients)
    if is_valid(poly):
        return poly
    return [1] * 12
# EVOLVE-BLOCK-END
