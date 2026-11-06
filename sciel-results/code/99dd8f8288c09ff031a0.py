import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Fallback: (1 + x + ... + x^11)^2 has 21 non-zero terms
    fallback = [1] * 12
    fallback_score = 21

    best_score = fallback_score
    best_candidate = fallback

    def compute_score_fast(exps, coeffs):
        """
        Fast computation of nonzero terms in P(x)^2.
        exps: list of 12 exponents (sorted)
        coeffs: list of 12 coefficients
        Returns count of nonzero terms, or inf if invalid.
        """
        if not exps or not coeffs:
            return float('inf')

        max_deg = max(exps)
        if max_deg > 40:
            return float('inf')
        if any(abs(c) > 64 for c in coeffs):
            return float('inf')
        if any(c == 0 for c in coeffs):
            return float('inf')
        if len(set(exps)) != 12:
            return float('inf')

        # Build dense array
        p = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in zip(exps, coeffs):
            p[e] = c

        q = np.convolve(p, p)
        count = int(np.count_nonzero(q))
        return count

    def _update_best(exps, coeffs, score):
        """Update the global best candidate if improved."""
        nonlocal best_score, best_candidate
        if score < best_score:
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            if len(dense) <= 41 and dense[-1] != 0:
                best_score = score
                best_candidate = dense

    def simulated_annealing(exps, coeffs, max_iters=2000):
        """
        Simulated annealing with diverse moves:
        1. Flip sign of one coefficient
        2. Shift one exponent by +/-1 to +/-3
        3. Move two exponents together (preserve relative offset)
        4. Jump: random reposition of one exponent
        """
        nonlocal best_score, best_candidate

        # Ensure sorted
        idx = np.argsort(exps)
        exps = [exps[i] for i in idx]
        coeffs = [coeffs[i] for i in idx]

        current_score = compute_score_fast(exps, coeffs)
        _update_best(exps, coeffs, current_score)

        if current_score <= 12:
            return

        # Annealing parameters
        T = 5.0
        T_min = 0.1
        cooling = 0.995

        for it in range(max_iters):
            if time.time() > deadline - 3:
                break

            # Choose move type
            move_type = random.random()
            new_exps = exps[:]
            new_coeffs = coeffs[:]

            if move_type < 0.25:
                # Flip sign of one coefficient
                i = random.randint(0, 11)
                new_coeffs[i] = -new_coeffs[i]
            elif move_type < 0.5:
                # Shift one exponent by small amount
                i = random.randint(0, 11)
                delta = random.choice([-3, -2, -1, 1, 2, 3])
                new_e = new_exps[i] + delta
                if 0 <= new_e <= 40 and new_e not in new_exps:
                    new_exps[i] = new_e
                else:
                    continue
            elif move_type < 0.7:
                # Move two exponents together
                i, j = random.sample(range(12), 2)
                offset = new_exps[j] - new_exps[i]
                delta = random.choice([-3, -2, -1, 1, 2, 3])
                new_ei = new_exps[i] + delta
                new_ej = new_exps[j] + delta
                if 0 <= new_ei <= 40 and 0 <= new_ej <= 40:
                    # Check no collisions
                    temp = new_exps[:]
                    temp[i] = new_ei
                    temp[j] = new_ej
                    if len(set(temp)) == 12:
                        new_exps[i] = new_ei
                        new_exps[j] = new_ej
                    else:
                        continue
                else:
                    continue
            elif move_type < 0.85:
                # Jump: random reposition of one exponent
                i = random.randint(0, 11)
                new_e = random.randint(0, 40)
                if new_e not in new_exps:
                    new_exps[i] = new_e
                else:
                    continue
            else:
                # Change coefficient magnitude
                i = random.randint(0, 11)
                old_c = new_coeffs[i]
                # Try small magnitudes
                new_c = random.choice([-3, -2, -1, 1, 2, 3, 4, 5])
                if new_c != old_c and abs(new_c) <= 64:
                    new_coeffs[i] = new_c
                else:
                    continue

            # Sort new state
            nidx = np.argsort(new_exps)
            new_exps = [new_exps[i] for i in nidx]
            new_coeffs = [new_coeffs[i] for i in nidx]

            new_score = compute_score_fast(new_exps, new_coeffs)

            # Accept or reject
            if new_score < current_score:
                exps = new_exps
                coeffs = new_coeffs
                current_score = new_score
                _update_best(exps, coeffs, current_score)
                if current_score <= 12:
                    break
            else:
                # Simulated annealing acceptance
                delta_score = new_score - current_score
                if T > T_min:
                    accept_prob = np.exp(-delta_score / T)
                    if random.random() < accept_prob:
                        exps = new_exps
                        coeffs = new_coeffs
                        current_score = new_score

            T = max(T * cooling, T_min)

        return exps, coeffs, current_score

    def generate_initial_candidates():
        """Generate diverse initial candidates for multi-start."""
        candidates = []

        # 1. Consecutive exponents with various sign patterns
        for start in range(0, 6):
            exps = [start + i for i in range(12)]
            if max(exps) <= 40:
                patterns = [
                    [1] * 12,
                    [1 if i % 2 == 0 else -1 for i in range(12)],
                    [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
                    [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                    [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
                    [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
                ]
                for pattern in patterns:
                    candidates.append((exps[:], pattern[:]))

        # 2. Gapped exponents
        gap_patterns = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15],
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
            [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        ]
        for exps in gap_patterns:
            for coeffs in [[1] * 12, [1 if i % 2 == 0 else -1 for i in range(12)]]:
                if max(exps) <= 40:
                    candidates.append((exps[:], coeffs[:]))

        # 3. Random sparse exponents
        random.seed(123)
        for _ in range(10):
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            candidates.append((exps, coeffs))

        # 4. Clustered exponents (some close, some far)
        for _ in range(5):
            # Create clusters
            cluster_centers = random.sample(range(0, 41), 3)
            exps = []
            for center in cluster_centers:
                for offset in random.sample([-2, -1, 0, 1, 2], 4):
                    e = center + offset
                    if 0 <= e <= 40 and e not in exps:
                        exps.append(e)
            while len(exps) < 12:
                e = random.randint(0, 40)
                if e not in exps:
                    exps.append(e)
            exps = sorted(exps[:12])
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            candidates.append((exps, coeffs))

        return candidates

    # --- Main search loop ---

    candidates = generate_initial_candidates()

    # First pass: quick evaluation to find good starting points
    scored_candidates = []
    for exps, coeffs in candidates:
        if time.time() > deadline - 10:
            break
        score = compute_score_fast(exps, coeffs)
        scored_candidates.append((score, exps, coeffs))

    scored_candidates.sort(key=lambda x: x[0])

    # Run simulated annealing on top candidates
    num_starts = min(8, len(scored_candidates))
    for i in range(num_starts):
        if time.time() > deadline - 5:
            break

        _, exps, coeffs = scored_candidates[i]
        simulated_annealing(exps, coeffs, max_iters=1500)

        if best_score <= 12:
            break

    # Second pass: more aggressive search on the best found
    if time.time() < deadline - 15 and best_score > 12:
        # Restart from best candidate with more iterations
        # Extract exponents and coeffs from best_candidate
        best_exps = []
        best_coeffs = []
        for i, c in enumerate(best_candidate):
            if c != 0:
                best_exps.append(i)
                best_coeffs.append(c)

        if len(best_exps) == 12:
            simulated_annealing(best_exps, best_coeffs, max_iters=3000)

    # Third pass: random restarts if time remains
    if time.time() < deadline - 10 and best_score > 12:
        random.seed(456)
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            simulated_annealing(exps, coeffs, max_iters=1000)
            if best_score <= 12:
                break

    return best_candidate

# EVOLVE-BLOCK-END
