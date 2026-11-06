"""Combined: exhaustive sign search over structured exponents + SA refinement."""
# EVOLVE-BLOCK-START
import random
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 90s kill

    best_score = float('inf')
    best_poly = None

    def _score(poly):
        """Count nonzero terms in P^2 using numpy convolve."""
        if poly is None or len(poly) == 0:
            return float('inf')
        nz = [i for i in range(len(poly)) if poly[i] != 0]
        if len(nz) != 12:
            return float('inf')
        p = np.array(poly, dtype=np.int64)
        q = np.convolve(p, p)
        return int(np.count_nonzero(q))

    def _update(poly, score):
        nonlocal best_score, best_poly
        if score < best_score:
            best_score = score
            best_poly = poly[:]

    def _make_dense(exps, coeffs):
        max_deg = max(exps)
        dense = [0] * (max_deg + 1)
        for e, c in zip(exps, coeffs):
            dense[e] = c
        return dense

    def _exhaustive_signs(exps, mags, deadline):
        """Try all 2^12 sign patterns for fixed exponents and magnitudes."""
        n = 12
        max_deg = max(exps)
        base = np.zeros(max_deg + 1, dtype=np.int64)
        for e, m in zip(exps, mags):
            base[e] = m

        best_local = float('inf')
        best_coeffs = None

        # Vectorized: compute all 2048 sign patterns at once
        # For each sign pattern, P = sum(s_i * m_i * x^e_i)
        # P^2 = sum_i s_i^2 m_i^2 x^{2e_i} + 2 sum_{i<j} s_i s_j m_i m_j x^{e_i+e_j}
        # = sum_i m_i^2 x^{2e_i} + 2 sum_{i<j} s_i s_j m_i m_j x^{e_i+e_j}
        # The diagonal part is fixed. Only cross terms depend on signs.

        # Precompute: for each pair (i,j), i<j, the exponent e_i+e_j and coefficient 2*m_i*m_j
        # For each sign pattern, cross term at position k = 2 * sum_{i<j, e_i+e_j=k} s_i*s_j*m_i*m_j
        # Total at position k = (diagonal if k=2e_i for some i) + cross

        # Let's just do it directly but efficiently
        # For each of 2048 patterns, build the polynomial and convolve
        # But that's 2048 convolves of length ~41... might be slow
        # Better: precompute the contribution matrix

        # Actually, let's use the structure:
        # P^2[k] = sum_{i: 2e_i=k} m_i^2 + 2 * sum_{i<j: e_i+e_j=k} s_i*s_j*m_i*m_j
        # The first part is fixed. The second part is a linear combination of s_i*s_j.

        # For 12 terms, there are 66 pairs. Each pair contributes to one position.
        # For a given sign pattern, we can compute P^2 in O(12^2) = 144 ops.
        # 2048 * 144 = ~300K ops total. Very fast.

        # Precompute pair data
        pairs = []  # (i, j, e_i+e_j, 2*m_i*m_j)
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j, exps[i] + exps[j], 2 * mags[i] * mags[j]))

        # Diagonal contributions: for each i, position 2*e_i gets m_i^2
        diag = {}
        for i in range(n):
            pos = 2 * exps[i]
            diag[pos] = diag.get(pos, 0) + mags[i] * mags[i]

        # For each sign pattern, compute P^2
        for mask in range(2048):
            if time.time() > deadline - 2:
                break
            signs = [1 if (mask >> i) & 1 else -1 for i in range(n)]

            # Build P^2
            sq = {}
            # Diagonal
            for pos, val in diag.items():
                sq[pos] = val
            # Cross terms
            for (i, j, pos, coeff) in pairs:
                sq[pos] = sq.get(pos, 0) + coeff * signs[i] * signs[j]

            # Count nonzero
            count = sum(1 for v in sq.values() if v != 0)
            if count < best_local:
                best_local = count
                best_coeffs = [signs[i] * mags[i] for i in range(n)]

        if best_coeffs is not None:
            return best_local, best_coeffs
        return None

    def _sa_refine(poly, max_iter, t_start, t_end, deadline):
        """Simulated annealing to refine a candidate."""
        current = poly[:]
        current_score = _score(current)
        best = current[:]
        best_score = current_score
        temp = t_start

        for _ in range(max_iter):
            if time.time() > deadline - 2:
                break

            nz = [i for i in range(len(current)) if current[i] != 0]
            if len(nz) != 12:
                break

            move_type = random.random()
            old_state = None

            if move_type < 0.4:
                # Change coefficient value
                idx = random.choice(nz)
                old_val = current[idx]
                new_val = random.randint(1, 64)
                if random.random() < 0.5:
                    new_val = -new_val
                if new_val == old_val:
                    continue
                current[idx] = new_val
                old_state = (idx, old_val)

            elif move_type < 0.7:
                # Move a term to a new position
                idx = random.choice(nz)
                coeff = current[idx]
                found = False
                for _ in range(20):
                    new_pos = random.randint(0, 40)
                    if current[new_pos] == 0:
                        current[idx] = 0
                        current[new_pos] = coeff
                        old_state = (idx, new_pos, coeff)
                        found = True
                        break
                if not found:
                    continue

            else:
                # Swap two coefficients
                i1, i2 = random.sample(nz, 2)
                v1, v2 = current[i1], current[i2]
                current[i1], current[i2] = v2, v1
                old_state = (i1, i2, v1, v2)

            new_score = _score(current)

            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best = current[:]
            else:
                delta = new_score - current_score
                if delta > 0 and random.random() < math.exp(-delta / max(temp, 1e-10)):
                    current_score = new_score
                else:
                    if old_state:
                        if len(old_state) == 2:
                            idx, old_val = old_state
                            current[idx] = old_val
                        elif len(old_state) == 3:
                            idx, new_pos, coeff = old_state
                            current[new_pos] = 0
                            current[idx] = coeff
                        elif len(old_state) == 4:
                            i1, i2, v1, v2 = old_state
                            current[i1], current[i2] = v1, v2

            temp *= 0.998
            if temp < t_end:
                temp = t_end

        return best, best_score

    # Phase 1: Structured exponent sets with exhaustive sign search
    candidate_exps = []

    # Arithmetic progressions
    for a in range(0, 15):
        for d in range(1, 5):
            exps = [a + i * d for i in range(12)]
            if max(exps) <= 40:
                candidate_exps.append(tuple(exps))

    # Two blocks
    for k in range(2, 11):
        for a in range(0, 12):
            for b in range(a + k, 35):
                block1 = [a + i for i in range(k)]
                block2 = [b + i for i in range(12 - k)]
                if max(block2) <= 40:
                    exps = tuple(sorted(block1 + block2))
                    candidate_exps.append(exps)

    # Three blocks
    for k1 in range(2, 6):
        for k2 in range(2, 6):
            k3 = 12 - k1 - k2
            if k3 < 2:
                continue
            for a in range(0, 10):
                for b in range(a + k1 + 1, 25):
                    for c in range(b + k2 + 1, 38):
                        block1 = [a + i for i in range(k1)]
                        block2 = [b + i for i in range(k2)]
                        block3 = [c + i for i in range(k3)]
                        if max(block3) <= 40:
                            exps = tuple(sorted(block1 + block2 + block3))
                            candidate_exps.append(exps)

    # Random sets
    random.seed(42)
    for _ in range(80):
        exps = tuple(sorted(random.sample(range(0, 41), 12)))
        candidate_exps.append(exps)

    # Deduplicate
    unique_exps = list(set(candidate_exps))

    # Sort by spread (larger range = more potential for distinct cross-term positions)
    unique_exps.sort(key=lambda e: e[-1] - e[0], reverse=True)

    mags = [1] * 12

    for exps in unique_exps:
        if time.time() > deadline - 10:
            break
        result = _exhaustive_signs(list(exps), mags, deadline)
        if result:
            score, coeffs = result
            dense = _make_dense(list(exps), coeffs)
            _update(dense, score)
            if best_score <= 13:
                break

    # Phase 2: Try some magnitude variations on best exponent sets
    if best_score > 13 and time.time() < deadline - 8:
        # Get top few exponent sets by spread
        top_exps = unique_exps[:30]
        mag_patterns = [
            [1] * 12,
            [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
        ]
        for exps in top_exps:
            if time.time() > deadline - 5:
                break
            for mag in mag_patterns:
                result = _exhaustive_signs(list(exps), mag, deadline)
                if result:
                    score, coeffs = result
                    dense = _make_dense(list(exps), coeffs)
                    _update(dense, score)
                    if best_score <= 13:
                        break
            if best_score <= 13:
                break

    # Phase 3: SA refinement on the best candidate
    if best_poly is not None and time.time() < deadline - 3:
        best_poly, best_score = _sa_refine(best_poly, max_iter=20000, t_start=1.0, t_end=0.001, deadline=deadline)

    # Phase 4: A few random restarts with SA if time remains
    if time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline - 3:
                break
            positions = sorted(random.sample(range(0, 41), 12))
            poly = [0] * 41
            for pos in positions:
                c = random.randint(1, 64)
                if random.random() < 0.5:
                    c = -c
                poly[pos] = c

            poly, score = _sa_refine(poly, max_iter=5000, t_start=1.0, t_end=0.01, deadline=deadline)
            _update(poly, score)

    # Final: trim and validate
    if best_poly is None:
        # Fallback: alternating signs on 0..11
        fallback = [1 if i % 2 == 0 else -1 for i in range(12)] + [0] * 29
        return fallback

    # Trim trailing zeros
    last_nz = -1
    for i in range(len(best_poly) - 1, -1, -1):
        if best_poly[i] != 0:
            last_nz = i
            break

    if last_nz == -1:
        return [1] + [0] * 10 + [1]

    result = best_poly[:last_nz + 1]
    nz_count = sum(1 for c in result if c != 0)
    if nz_count != 12:
        fallback = [1 if i % 2 == 0 else -1 for i in range(12)] + [0] * 29
        return fallback

    return result

# EVOLVE-BLOCK-END
