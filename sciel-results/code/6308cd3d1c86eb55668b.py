import numpy as np
import time
import random
from itertools import product, combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82  # Safety margin before 90s kill

    def count_nonzero(poly):
        return int(np.count_nonzero(poly))

    def poly_square(coeffs):
        return np.convolve(coeffs, coeffs)

    def evaluate(coeffs):
        sq = poly_square(coeffs)
        return count_nonzero(sq)

    def is_valid(coeffs):
        if len(coeffs) != 12:
            return False
        if sum(1 for c in coeffs if c != 0) != 12:
            return False
        if any(c < -64 or c > 64 for c in coeffs):
            return False
        return True

    best_score = float('inf')
    best_poly = [1] * 12  # baseline

    def try_poly(coeffs):
        nonlocal best_score, best_poly
        if not is_valid(coeffs):
            return
        score = evaluate(coeffs)
        if score < best_score:
            best_score = score
            best_poly = list(coeffs)

    # Try baseline
    try_poly([1] * 12)

    # === PHASE 1: Structured initial candidates ===
    # Based on knowledge: structured position patterns (clusters, arithmetic progressions)
    # create systematic cross-term overlaps enabling more effective sign/magnitude assignments

    rng = random.Random(42)

    # Phase 1a: Arithmetic progression supports with sign patterns
    for restart in range(300):
        if time.time() > deadline:
            break
        pattern_type = restart % 8

        if pattern_type == 0:
            # Random signs, small magnitudes
            poly = [rng.choice([-1, 1]) * rng.randint(1, 3) for _ in range(12)]
        elif pattern_type == 1:
            # Palindromic with sign flips
            half = [rng.choice([-2, -1, 1, 2]) for _ in range(6)]
            poly = half + [h * rng.choice([-1, 1]) for h in half[::-1]]
        elif pattern_type == 2:
            # Two blocks with opposite signs
            k = rng.randint(4, 8)
            s = rng.choice([-1, 1])
            poly = [s * rng.randint(1, 3) for _ in range(k)] + [-s * rng.randint(1, 3) for _ in range(12 - k)]
        elif pattern_type == 3:
            # Arithmetic progression of coefficients
            a0 = rng.randint(-3, 3)
            d = rng.randint(-2, 2)
            if a0 == 0 and d == 0:
                a0 = 1
            poly = [a0 + d * i for i in range(12)]
            # Ensure no zeros
            for i in range(12):
                if poly[i] == 0:
                    poly[i] = 1
        elif pattern_type == 4:
            # Sparse magnitudes from {-4,-3,-2,-1,1,2,3,4}
            poly = [rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for _ in range(12)]
        elif pattern_type == 5:
            # Alternating sign with varying magnitude
            poly = [rng.choice([-1, 1]) * (1 + (i % 3)) * rng.randint(1, 2) for i in range(12)]
        elif pattern_type == 6:
            # Clustered: groups of similar values
            groups = [rng.randint(1, 3) for _ in range(4)]
            signs = [rng.choice([-1, 1]) for _ in range(4)]
            vals = [rng.randint(1, 4) for _ in range(4)]
            poly = []
            for g in range(4):
                poly.extend([signs[g] * vals[g]] * groups[g])
            if len(poly) != 12:
                poly = poly[:12] + [1] * (12 - len(poly))
            for i in range(12):
                if poly[i] == 0:
                    poly[i] = 1
        else:
            # Fibonacci-like or other structured
            base = rng.choice([-2, -1, 1, 2])
            poly = []
            for i in range(12):
                val = base * (1 + (i % 4)) * rng.choice([-1, 1])
                if val == 0:
                    val = 1
                poly.append(val)

        try_poly(poly)

    # === PHASE 2: Local search with multiple coordinate changes ===
    def local_search(coeffs, max_iters=2000, seed=42):
        """Local search trying single and double coordinate changes."""
        rng_ls = random.Random(seed)
        curr = list(coeffs)
        curr_score = evaluate(curr)
        best = list(curr)
        best_s = curr_score

        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Choose move type
            if rng_ls.random() < 0.7:
                # Single coordinate change
                pos = rng_ls.randint(0, 11)
                old_val = curr[pos]

                # Generate smart candidates
                candidates = set()
                # Based on neighboring coefficients
                for offset in [-2, -1, 1, 2]:
                    nb = (pos + offset) % 12
                    v = curr[nb]
                    for mult in [-3, -2, -1, 1, 2, 3]:
                        candidates.add(v * mult)
                # Small random values
                for _ in range(4):
                    candidates.add(rng_ls.randint(-8, 8))
                # Specific values
                candidates.update([-1, 1, -2, 2, -3, 3, -4, 4])

                candidates = [v for v in candidates if -64 <= v <= 64 and v != old_val and v != 0]
                if not candidates:
                    continue

                new_val = rng_ls.choice(candidates)
                curr[pos] = new_val
                new_score = evaluate(curr)

                if new_score <= curr_score:
                    curr_score = new_score
                    if new_score < best_s:
                        best_s = new_score
                        best = list(curr)
                else:
                    # Accept worse with small probability
                    if rng_ls.random() < 0.03 and new_score <= curr_score + 2:
                        curr_score = new_score
                    else:
                        curr[pos] = old_val
            else:
                # Double coordinate change
                pos1, pos2 = rng_ls.sample(range(12), 2)
                old1, old2 = curr[pos1], curr[pos2]

                cands1 = [rng_ls.randint(-6, 6) for _ in range(5)] + [old1 * 2, old1 // 2 if old1 % 2 == 0 else old1]
                cands2 = [rng_ls.randint(-6, 6) for _ in range(5)] + [old2 * 2, old2 // 2 if old2 % 2 == 0 else old2]

                for nv1 in cands1:
                    for nv2 in cands2:
                        if nv1 == old1 or nv2 == old2 or nv1 == 0 or nv2 == 0:
                            continue
                        if not (-64 <= nv1 <= 64 and -64 <= nv2 <= 64):
                            continue
                        curr[pos1], curr[pos2] = nv1, nv2
                        new_score = evaluate(curr)
                        if new_score <= curr_score:
                            curr_score = new_score
                            if new_score < best_s:
                                best_s = new_score
                                best = list(curr)
                            break
                    else:
                        continue
                    break
                else:
                    curr[pos1], curr[pos2] = old1, old2

        return best, best_s

    # Run local search from best found so far
    if time.time() < deadline:
        best_poly, best_score = local_search(best_poly, max_iters=3000, seed=1)
        try_poly(best_poly)

    # === PHASE 3: Constraint-based search with zero patterns ===
    # For P(x)^2, c_0 = a_0^2 > 0 and c_22 = a_11^2 > 0 always
    # Try to zero out interior coefficients

    def get_pairs(k):
        pairs = []
        for i in range(max(0, k - 11), min(11, k) + 1):
            j = k - i
            if 0 <= j <= 11:
                pairs.append((i, j))
        return pairs

    pair_map = {k: get_pairs(k) for k in range(23)}

    def solve_with_zero_constraint(zero_set, max_iters=4000, seed=42):
        """Local search to find a 12-term poly whose square vanishes on zero_set."""
        rng_zs = random.Random(seed)
        # Start with random coefficients
        coeffs = [rng_zs.choice([-1, 1]) * rng_zs.randint(1, 6) for _ in range(12)]

        def eval_with_zeros(c):
            sq = poly_square(c)
            nz = count_nonzero(sq)
            violations = sum(1 for k in zero_set if sq[k] != 0)
            return nz, violations

        curr_nz, curr_viol = eval_with_zeros(coeffs)
        best_viol = curr_viol
        best_nz = curr_nz
        best_coeffs = list(coeffs)

        for it in range(max_iters):
            if time.time() > deadline:
                break

            pos = rng_zs.randint(0, 11)
            old_val = coeffs[pos]

            # Generate candidates biased toward creating cancellations
            candidates = set()
            for other in range(12):
                if other != pos:
                    v = coeffs[other]
                    for mult in [-3, -2, -1, 1, 2, 3]:
                        candidates.add(v * mult)
                    candidates.add(0)
            for _ in range(6):
                candidates.add(rng_zs.randint(-12, 12))

            candidates = [v for v in candidates if -64 <= v <= 64 and v != old_val]
            if not candidates:
                continue

            new_val = rng_zs.choice(candidates)
            coeffs[pos] = new_val

            # Ensure exactly 12 nonzero
            nz_count = sum(1 for c in coeffs if c != 0)
            if nz_count != 12:
                if nz_count < 12:
                    zero_positions = [i for i in range(12) if coeffs[i] == 0]
                    if zero_positions:
                        zp = rng_zs.choice(zero_positions)
                        coeffs[zp] = rng_zs.choice([-1, 1]) * rng_zs.randint(1, 4)
                        nz_count = sum(1 for c in coeffs if c != 0)
                if nz_count != 12:
                    coeffs[pos] = old_val
                    continue

            new_nz, new_viol = eval_with_zeros(coeffs)
            if (new_viol, new_nz) < (curr_viol, curr_nz):
                curr_viol = new_viol
                curr_nz = new_nz
                if (new_viol, new_nz) < (best_viol, best_nz):
                    best_viol = new_viol
                    best_nz = new_nz
                    best_coeffs = list(coeffs)
            else:
                if rng_zs.random() < 0.04 and new_viol <= curr_viol + 2:
                    curr_viol = new_viol
                    curr_nz = new_nz
                else:
                    coeffs[pos] = old_val

        return best_coeffs, best_viol, best_nz

    # Try various zero-set patterns
    # Focus on patterns that are algebraically plausible
    zero_set_candidates = []

    # Small zero sets
    for size in [1, 2, 3]:
        count = 0
        for k in range(1, 22):
            if size == 1:
                zero_set_candidates.append(frozenset([k]))
                count += 1
            elif size == 2:
                for j in range(k + 1, 22):
                    zero_set_candidates.append(frozenset([k, j]))
                    count += 1
                    if count > 300:
                        break
            else:
                for j in range(k + 1, 22):
                    for l in range(j + 1, 22):
                        zero_set_candidates.append(frozenset([k, j, l]))
                        count += 1
                        if count > 300:
                            break
                    if count > 300:
                        break
            if count > 300:
                break
        if count > 300:
            break

    # Structured patterns
    structured = [
        frozenset(range(1, 22, 2)),   # odd indices
        frozenset(range(2, 22, 2)),   # even indices
        frozenset(range(1, 11)),      # first half interior
        frozenset(range(12, 22)),     # second half interior
        frozenset(range(2, 21, 3)),   # every 3rd
        frozenset(range(1, 22, 4)),   # every 4th
        frozenset(range(3, 20, 3)),   # every 3rd offset
    ]
    zero_set_candidates.extend(structured)

    random.shuffle(zero_set_candidates)

    for zs in zero_set_candidates[:80]:
        if time.time() > deadline:
            break

        target_nz = 23 - len(zs)
        if best_score <= target_nz:
            continue

        for seed in [42, 123, 777]:
            if time.time() > deadline:
                break
            coeffs, viol, nz = solve_with_zero_constraint(zs, max_iters=2500, seed=seed)
            if coeffs is not None and is_valid(coeffs):
                actual_nz = evaluate(coeffs)
                if actual_nz < best_score:
                    best_score = actual_nz
                    best_poly = list(coeffs)

                # Local refinement
                if time.time() < deadline:
                    best_poly, best_score = local_search(best_poly, max_iters=1000, seed=seed)
                    try_poly(best_poly)

    # === PHASE 4: Exhaustive single-coordinate refinement ===
    if time.time() < deadline:
        for pos in range(12):
            if time.time() > deadline:
                break
            old_val = best_poly[pos]
            for new_val in range(-64, 65):
                if new_val == old_val or new_val == 0:
                    continue
                best_poly[pos] = new_val
                new_score = evaluate(best_poly)
                if new_score < best_score:
                    best_score = new_score
                else:
                    best_poly[pos] = old_val
            best_poly[pos] = old_val

    # === PHASE 5: Second refinement pass with wider range ===
    if time.time() < deadline:
        for pos in range(12):
            if time.time() > deadline:
                break
            old_val = best_poly[pos]
            for new_val in range(-64, 65):
                if new_val == old_val or new_val == 0:
                    continue
                best_poly[pos] = new_val
                new_score = evaluate(best_poly)
                if new_score < best_score:
                    best_score = new_score
                else:
                    best_poly[pos] = old_val
            best_poly[pos] = old_val

    # === PHASE 6: Final local search polish ===
    if time.time() < deadline:
        best_poly, best_score = local_search(best_poly, max_iters=2000, seed=999)
        try_poly(best_poly)

    return best_poly
