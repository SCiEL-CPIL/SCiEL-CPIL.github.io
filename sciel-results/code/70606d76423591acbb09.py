"""Search for sparse squares of 12-term integer polynomials."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 82.0  # safety margin before 90s kill
    rng = random.Random(42)

    best_coeffs = None
    best_score = 144  # upper bound: 12*12 = 144 terms if no cancellation

    def count_nonzero_square(coeffs):
        """Count nonzero terms in the exact integer square using numpy bincount."""
        nz = np.nonzero(coeffs)[0]
        if len(nz) == 0:
            return 0
        c = coeffs[nz]
        # All pairwise products
        i, j = np.meshgrid(nz, nz, indexing='ij')
        sums = i + j
        prods = c[:, None] * c[None, :]
        # Use bincount to sum products by degree
        max_deg = int(sums.max())
        if max_deg < 0:
            return 0
        sq = np.bincount(sums, weights=prods, minlength=max_deg + 1)
        return int(np.count_nonzero(sq))

    def try_candidate(coeffs):
        """Check validity and return (valid, score)."""
        if len(coeffs) > 41:
            return False, 10**9
        if coeffs[-1] == 0:
            # trim trailing zeros
            while len(coeffs) > 1 and coeffs[-1] == 0:
                coeffs = coeffs[:-1]
        if len(coeffs) > 41:
            return False, 10**9
        if any(abs(c) > 64 for c in coeffs):
            return False, 10**9
        nz_count = sum(1 for c in coeffs if c != 0)
        if nz_count != 12:
            return False, 10**9
        if coeffs[-1] == 0:
            return False, 10**9
        score = count_nonzero_square(coeffs)
        return True, score

    def local_search(coeffs, deadline):
        """Local search: try flipping signs, changing magnitudes, shifting exponents."""
        nonlocal best_coeffs, best_score
        cur = list(coeffs)
        cur_score = count_nonzero_square(cur)
        if cur_score < best_score:
            best_score = cur_score
            best_coeffs = list(cur)

        max_iters = 500
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if len(cur) > 41:
                break
            improved = False

            # Strategy 1: flip sign of a random nonzero term
            nz_idx = [i for i, c in enumerate(cur) if c != 0]
            if nz_idx:
                idx = rng.choice(nz_idx)
                cur[idx] = -cur[idx]
                new_score = count_nonzero_square(cur)
                if new_score < cur_score:
                    cur_score = new_score
                    improved = True
                    if cur_score < best_score:
                        best_score = cur_score
                        best_coeffs = list(cur)
                else:
                    cur[idx] = -cur[idx]  # revert

            # Strategy 2: change magnitude of a random nonzero term
            if nz_idx:
                idx = rng.choice(nz_idx)
                old_val = cur[idx]
                choices = [v for v in range(-64, 65) if v != 0 and v != old_val and abs(v) <= 64]
                if choices:
                    cur[idx] = rng.choice(choices)
                    new_score = count_nonzero_square(cur)
                    if new_score < cur_score:
                        cur_score = new_score
                        improved = True
                        if cur_score < best_score:
                            best_score = cur_score
                            best_coeffs = list(cur)
                    else:
                        cur[idx] = old_val

            # Strategy 3: shift a random nonzero term's exponent by ±1
            if nz_idx and len(cur) < 41:
                idx = rng.choice(nz_idx)
                old_val = cur[idx]
                shift = rng.choice([-1, 1])
                new_idx = idx + shift
                if 0 <= new_idx < 41 and (new_idx >= len(cur) or cur[new_idx] == 0):
                    # Extend if needed
                    while len(cur) <= new_idx:
                        cur.append(0)
                    cur[new_idx] = old_val
                    cur[idx] = 0
                    new_score = count_nonzero_square(cur)
                    if new_score < cur_score:
                        cur_score = new_score
                        improved = True
                        if cur_score < best_score:
                            best_score = cur_score
                            best_coeffs = list(cur)
                    else:
                        cur[idx] = old_val
                        cur[new_idx] = 0

            # Strategy 4: swap exponents of two nonzero terms
            if len(nz_idx) >= 2:
                i, j = rng.sample(nz_idx, 2)
                cur[i], cur[j] = cur[j], cur[i]
                new_score = count_nonzero_square(cur)
                if new_score < cur_score:
                    cur_score = new_score
                    improved = True
                    if cur_score < best_score:
                        best_score = cur_score
                        best_coeffs = list(cur)
                else:
                    cur[i], cur[j] = cur[j], cur[i]

            if not improved:
                # Try a more aggressive perturbation
                if nz_idx:
                    idx = rng.choice(nz_idx)
                    old_val = cur[idx]
                    cur[idx] = 0
                    if len(cur) < 41:
                        new_idx = rng.randint(0, min(40, len(cur) - 1 + 1))
                        while len(cur) <= new_idx:
                            cur.append(0)
                        if cur[new_idx] == 0:
                            cur[new_idx] = rng.choice([-1, 1]) * rng.randint(1, 64)
                    new_score = count_nonzero_square(cur)
                    if new_score < cur_score:
                        cur_score = new_score
                        if cur_score < best_score:
                            best_score = cur_score
                            best_coeffs = list(cur)
                    else:
                        cur[idx] = old_val
                        if new_idx < len(cur):
                            cur[new_idx] = 0

        return cur_score

    # Phase 1: Structured random generation with sign patterns
    # Try various exponent patterns that promote cancellation
    patterns = []

    # Pattern A: Two arithmetic progressions
    for d1 in [1, 2, 3]:
        for d2 in [1, 2, 3]:
            for offset in range(0, 15):
                exps = []
                for i in range(6):
                    exps.append(i * d1)
                for i in range(6):
                    exps.append(offset + i * d2)
                if len(set(exps)) == 12:
                    patterns.append(sorted(exps))

    # Pattern B: Three clusters
    for c1 in [1, 2, 3]:
        for c2 in [1, 2, 3]:
            for c3 in [1, 2, 3]:
                for o1 in range(0, 10):
                    for o2 in range(0, 10):
                        for o3 in range(0, 10):
                            exps = []
                            for i in range(4):
                                exps.append(o1 + i * c1)
                            for i in range(4):
                                exps.append(o2 + i * c2)
                            for i in range(4):
                                exps.append(o3 + i * c3)
                            if len(set(exps)) == 12 and max(exps) < 41:
                                patterns.append(sorted(exps))

    # Pattern C: Random sparse exponents
    for _ in range(200):
        exps = sorted(rng.sample(range(0, 41), 12))
        patterns.append(exps)

    # Deduplicate
    seen = set()
    unique_patterns = []
    for p in patterns:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)

    # Evaluate patterns with various sign/coefficient assignments
    trials_per_pattern = 5
    for exps in unique_patterns:
        if time.time() > deadline:
            break
        for trial in range(trials_per_pattern):
            if time.time() > deadline:
                break
            coeffs = [0] * (max(exps) + 1)
            for e in exps:
                sign = rng.choice([-1, 1])
                mag = rng.choice([1, 1, 1, 2, 2, 3, 5, 8, 13, 21, 34])
                coeffs[e] = sign * mag
            valid, score = try_candidate(coeffs)
            if valid and score < best_score:
                best_score = score
                best_coeffs = list(coeffs)
                local_search(coeffs, deadline)

    # Phase 2: Pure random search
    random_trials = 5000
    for _ in range(random_trials):
        if time.time() > deadline:
            break
        exps = sorted(rng.sample(range(0, 41), 12))
        coeffs = [0] * (max(exps) + 1)
        for e in exps:
            sign = rng.choice([-1, 1])
            mag = rng.choice([1, 1, 1, 2, 2, 3, 5, 8, 13, 21, 34, 55])
            coeffs[e] = sign * mag
        valid, score = try_candidate(coeffs)
        if valid and score < best_score:
            best_score = score
            best_coeffs = list(coeffs)
            local_search(coeffs, deadline)

    # Phase 3: Targeted sign-flip search on best found
    if best_coeffs is not None and time.time() < deadline:
        for _ in range(200):
            if time.time() > deadline:
                break
            cur = list(best_coeffs)
            nz_idx = [i for i, c in enumerate(cur) if c != 0]
            # Flip 1-3 random signs
            k = rng.randint(1, min(3, len(nz_idx)))
            for idx in rng.sample(nz_idx, k):
                cur[idx] = -cur[idx]
            score = count_nonzero_square(cur)
            if score < best_score:
                best_score = score
                best_coeffs = list(cur)
                local_search(cur, deadline)

    # Final validation
    if best_coeffs is None:
        # Fallback: 12 ones at consecutive positions
        best_coeffs = [1] * 12
        best_score = count_nonzero_square(best_coeffs)

    # Ensure validity
    valid, score = try_candidate(best_coeffs)
    if not valid:
        best_coeffs = [1] * 12
        best_score = count_nonzero_square(best_coeffs)

    return best_coeffs
# EVOLVE-BLOCK-END
