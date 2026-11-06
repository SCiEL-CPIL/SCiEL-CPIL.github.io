import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Fallback: [1]*12 gives square with 21 nonzero terms
    fallback = [1] * 12
    best_score = 21
    best_candidate = fallback

    def square_nonzero_count_fast(exponents, coefficients):
        """Compute P^2 nonzero count using numpy convolution."""
        max_exp = max(exponents)
        dense = np.zeros(max_exp + 1, dtype=np.int64)
        for e, c in zip(exponents, coefficients):
            dense[e] = c
        sq = np.convolve(dense, dense)
        return int(np.count_nonzero(sq))

    def build_coeff_list(exponents, coefficients):
        max_exp = max(exponents)
        if max_exp > 40:
            return None
        result = [0] * (max_exp + 1)
        for e, c in zip(exponents, coefficients):
            result[e] = c
        if result[-1] == 0:
            return None
        if any(abs(x) > 64 for x in result):
            return None
        if len(result) > 41:
            return None
        nonzero = sum(1 for x in result if x != 0)
        if nonzero != 12:
            return None
        return result

    def pair_sum_collapse(exps):
        """Count how many pair sums collide. Higher = more potential cancellation."""
        n = len(exps)
        sums = defaultdict(int)
        for i in range(n):
            for j in range(i + 1, n):
                sums[exps[i] + exps[j]] += 1
        # Number of "lost" terms due to collisions
        total_pairs = n * (n - 1) // 2
        distinct_sums = len(sums)
        return total_pairs - distinct_sums

    # Strategy: enumerate structured exponent sets, then do coefficient search
    # Key insight: for a 12-term poly, P^2 has at most 12*13/2 = 78 terms before cancellation
    # We need to minimize nonzero terms in P^2.
    # With all coefficients ±1, P^2 has 12 self-terms (always nonzero) + 66 cross-terms (pairs)
    # If pair sums collide, cross-terms can cancel.
    
    # Generate diverse exponent sets
    exp_sets = []
    seen = set()

    # 1. Arithmetic progressions (most collisions)
    for start in range(0, 12):
        for step in range(1, 5):
            exps = tuple(start + i * step for i in range(12))
            if max(exps) <= 40 and exps not in seen:
                seen.add(exps)
                exp_sets.append(list(exps))

    # 2. Two interleaved APs
    for s1 in range(0, 6):
        for s2 in range(5, 18):
            for step in range(1, 4):
                e1 = [s1 + i * step for i in range(6)]
                e2 = [s2 + i * step for i in range(6)]
                exps = tuple(sorted(e1 + e2))
                if len(set(exps)) == 12 and max(exps) <= 40 and exps not in seen:
                    seen.add(exps)
                    exp_sets.append(list(exps))

    # 3. Three interleaved APs of 4
    for s1 in range(0, 5):
        for s2 in range(4, 14):
            for s3 in range(12, 28):
                for step in range(1, 3):
                    e1 = [s1 + i * step for i in range(4)]
                    e2 = [s2 + i * step for i in range(4)]
                    e3 = [s3 + i * step for i in range(4)]
                    exps = tuple(sorted(e1 + e2 + e3))
                    if len(set(exps)) == 12 and max(exps) <= 40 and exps not in seen:
                        seen.add(exps)
                        exp_sets.append(list(exps))

    # 4. Symmetric: pairs summing to constant
    for total in range(12, 41):
        for a in range(0, total // 2):
            for b in range(a + 1, total // 2):
                for c in range(b + 1, total // 2):
                    exps = tuple(sorted([a, b, c, total - a, total - b, total - c]))
                    if len(set(exps)) == 6:
                        # Expand to 12 by adding another symmetric set
                        for d in range(c + 1, total // 2):
                            exps12 = tuple(sorted([a, b, c, d, total - a, total - b, total - c, total - d]))
                            if len(set(exps12)) == 8 and max(exps12) <= 40:
                                # Add 4 more
                                for e in range(d + 1, total // 2):
                                    for f in range(e + 1, total // 2):
                                        exps12f = tuple(sorted([a, b, c, d, e, f, total - a, total - b, total - c, total - d, total - e, total - f]))
                                        if len(set(exps12f)) == 12 and max(exps12f) <= 40 and exps12f not in seen:
                                            seen.add(exps12f)
                                            exp_sets.append(list(exps12f))
                    if time.time() > deadline - 30:
                        break
                if time.time() > deadline - 30:
                    break
            if time.time() > deadline - 30:
                break
        if time.time() > deadline - 30:
            break

    # 5. Quadratic
    for a in range(1, 3):
        for b in range(0, 8):
            for c in range(0, 15):
                exps = tuple(sorted(a * i * i + b * i + c for i in range(12)))
                if len(set(exps)) == 12 and max(exps) <= 40 and min(exps) >= 0 and exps not in seen:
                    seen.add(exps)
                    exp_sets.append(list(exps))

    # 6. Geometric-like (powers of 2, 3)
    for base in [2, 3]:
        for offset in range(0, 5):
            exps = tuple(offset + base ** i for i in range(12))
            if max(exps) <= 40 and exps not in seen:
                seen.add(exps)
                exp_sets.append(list(exps))

    # 7. Mixed clusters
    for s1 in range(0, 5):
        for step1 in range(1, 3):
            for s2 in range(8, 18):
                for step2 in range(1, 3):
                    for s3 in range(20, 32):
                        for step3 in range(1, 3):
                            e1 = [s1 + i * step1 for i in range(4)]
                            e2 = [s2 + i * step2 for i in range(4)]
                            e3 = [s3 + i * step3 for i in range(4)]
                            exps = tuple(sorted(e1 + e2 + e3))
                            if len(set(exps)) == 12 and max(exps) <= 40 and exps not in seen:
                                seen.add(exps)
                                exp_sets.append(list(exps))

    # Sort by collapse potential (more collisions = better)
    exp_sets.sort(key=lambda e: -pair_sum_collapse(e))

    # Coefficient search: try small coefficient values
    # For each exponent set, search over coefficient assignments
    coeff_values = [1, -1, 2, -2, 3, -3]

    def search_coeffs(exps, max_evals=2000):
        """Search coefficient assignments for given exponents."""
        n = len(exps)
        best_local = best_score
        best_coeffs_local = None
        evals = 0

        # Start with all +1
        coeffs = [1] * n
        current = square_nonzero_count_fast(exps, coeffs)
        if current < best_local:
            best_local = current
            best_coeffs_local = coeffs[:]
            evals += 1

        # Greedy: try changing one coefficient at a time
        improved = True
        while improved and evals < max_evals and time.time() < deadline:
            improved = False
            for i in range(n):
                old = coeffs[i]
                for nv in coeff_values:
                    if nv == old:
                        continue
                    coeffs[i] = nv
                    evals += 1
                    sc = square_nonzero_count_fast(exps, coeffs)
                    if sc < best_local:
                        best_local = sc
                        best_coeffs_local = coeffs[:]
                        improved = True
                        break
                    elif sc < current:
                        # Accept if better than current but not best
                        current = sc
                        improved = True
                        break
                    else:
                        coeffs[i] = old
                else:
                    continue
                break
            else:
                if not improved:
                    break

        # Random restarts
        import random
        rng = random.Random(42)
        for _ in range(50):
            if time.time() > deadline:
                break
            coeffs = [rng.choice(coeff_values) for _ in range(n)]
            evals += 1
            sc = square_nonzero_count_fast(exps, coeffs)
            if sc < best_local:
                best_local = sc
                best_coeffs_local = coeffs[:]

        return best_local, best_coeffs_local

    # Process exponent sets
    for exps in exp_sets[:300]:
        if time.time() > deadline - 5:
            break

        # Quick check: what's the minimum possible?
        # 12 self-terms are always nonzero (coefficient squared > 0)
        # So minimum is 12. But realistically with 12 terms, cross-terms add more.
        # Skip if collapse is too low
        collapse = pair_sum_collapse(exps)
        if collapse < 20:
            continue

        sc, coeffs = search_coeffs(exps, max_evals=1500)

        if sc < best_score:
            best_score = sc
            candidate = build_coeff_list(exps, coeffs)
            if candidate is not None:
                best_candidate = candidate

    # Special: try factorization P = A * B where A, B have few terms
    # If P = A*B with A having a terms and B having b terms, P has at most a*b terms
    # For P to have 12 terms: (a,b) could be (1,12),(2,6),(3,4),(4,3),(6,2),(12,1)
    # P^2 = A^2 * B^2. If A^2 and B^2 have few nonzero terms, P^2 might too.
    if time.time() < deadline - 10:
        # Try P = (sum of 2 terms) * (sum of 6 terms)
        # A = x^a + x^b, B = sum of 6 terms
        # P has at most 12 terms (if no collisions)
        # P^2 = (x^a + x^b)^2 * B^2 = (x^{2a} + 2x^{a+b} + x^{2b}) * B^2
        # If B^2 has k nonzero terms, P^2 has at most 3k terms (shifted and summed)
        # To minimize: want B^2 to have few nonzero terms
        
        # B with 6 terms: B^2 has 6 self + 15 cross = 21 max
        # With sign optimization, can reduce
        # P^2 = x^{2a}*B^2 + 2x^{a+b}*B^2 + x^{2b}*B^2
        # If shifts don't collide, P^2 has 3*nonzero(B^2) terms
        # If shifts DO collide (e.g., 2a = a+b => a=b, not allowed for distinct)
        # Actually 2a, a+b, 2b are distinct if a != b
        # So P^2 nonzero = 3 * nonzero(B^2) unless cross-cancellation between the three shifted copies
        
        # Better: P = A * B where A has 3 terms, B has 4 terms
        # A^2 has 3 self + 3 cross = 6 max
        # B^2 has 4 self + 6 cross = 10 max
        # P^2 = A^2 * B^2: at most 6*10 = 60 terms, but with structure fewer
        
        # Try: A = 1 + x^a + x^b, B = 1 + x^c + x^d + x^e
        # P = (1 + x^a + x^b)(1 + x^c + x^d + x^e)
        # P has 12 terms if all sums distinct
        # P^2 = A^2 * B^2
        # A^2 = 1 + 2x^a + 2x^b + x^{2a} + 2x^{a+b} + x^{2b} (6 terms)
        # B^2 = 1 + 2x^c + 2x^d + 2x^e + x^{2c} + x^{2d} + x^{2e} + 2x^{c+d} + 2x^{c+e} + 2x^{d+e} (10 terms)
        # P^2 = A^2 * B^2: up to 60 terms, but many may cancel or coincide

        # Try specific small cases
        for a in range(1, 10):
            for b in range(a + 1, 15):
                for c in range(1, 8):
                    for d in range(c + 1, 12):
                        for e in range(d + 1, 16):
                            if time.time() > deadline - 3:
                                break
                            # A = 1 + x^a + x^b, B = 1 + x^c + x^d + x^e
                            # P = A * B
                            A_exps = [0, a, b]
                            B_exps = [0, c, d, e]
                            P_exps = []
                            P_coeffs = []
                            for ea in A_exps:
                                for eb in B_exps:
                                    P_exps.append(ea + eb)
                                    P_coeffs.append(1)
                            if len(set(P_exps)) != 12:
                                continue
                            if max(P_exps) > 40:
                                continue
                            # Sort
                            pairs = sorted(zip(P_exps, P_coeffs))
                            P_exps = [p[0] for p in pairs]
                            P_coeffs = [p[1] for p in pairs]
                            sc = square_nonzero_count_fast(P_exps, P_coeffs)
                            if sc < best_score:
                                best_score = sc
                                candidate = build_coeff_list(P_exps, P_coeffs)
                                if candidate is not None:
                                    best_candidate = candidate
                        if time.time() > deadline - 3:
                            break
                    if time.time() > deadline - 3:
                        break
                if time.time() > deadline - 3:
                    break
            if time.time() > deadline - 3:
                break
        if time.time() > deadline - 3:
            break

    # Try sign patterns on best exponent sets more aggressively
    if time.time() < deadline - 5:
        # Focus on top exponent sets by collapse
        top_exps = exp_sets[:50]
        for exps in top_exps:
            if time.time() > deadline - 2:
                break
            # Try all 2^12 sign patterns for the first few
            # But 2^12 = 4096, too many. Use smarter search.
            # Try flipping subsets of size 1, 2, 3 from all-ones
            base_coeffs = [1] * 12
            base_sc = square_nonzero_count_fast(exps, base_coeffs)
            if base_sc < best_score:
                best_score = base_sc
                candidate = build_coeff_list(exps, base_coeffs)
                if candidate is not None:
                    best_candidate = candidate

            # Flip 2 signs
            for i in range(12):
                for j in range(i + 1, 12):
                    if time.time() > deadline - 2:
                        break
                    coeffs = [1] * 12
                    coeffs[i] = -1
                    coeffs[j] = -1
                    sc = square_nonzero_count_fast(exps, coeffs)
                    if sc < best_score:
                        best_score = sc
                        candidate = build_coeff_list(exps, coeffs)
                        if candidate is not None:
                            best_candidate = candidate

    if best_score < 21:
        return best_candidate
    return fallback
