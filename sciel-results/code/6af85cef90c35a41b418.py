import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    best_score = 21
    best_candidate = [1] * 12  # fallback: (1+x+...+x^11)^2 has 21 nonzero terms

    def compute_square_nonzeros(exps, coeffs):
        """Compute number of nonzero terms in P(x)^2."""
        max_deg = max(exps)
        p = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in zip(exps, coeffs):
            p[e] = c
        q = np.convolve(p, p)
        return int(np.count_nonzero(q))

    def get_dense(exps, coeffs):
        max_deg = max(exps)
        dense = [0] * (max_deg + 1)
        for e, c in zip(exps, coeffs):
            dense[e] = c
        if dense[-1] != 0:
            return dense
        return None

    def try_update(exps, coeffs):
        nonlocal best_score, best_candidate
        if len(exps) != 12 or len(coeffs) != 12:
            return
        if len(set(exps)) != 12:
            return
        if max(exps) > 40 or min(exps) < 0:
            return
        if any(abs(c) > 64 or c == 0 for c in coeffs):
            return
        score = compute_square_nonzeros(exps, coeffs)
        if score < best_score:
            best_score = score
            cand = get_dense(exps, coeffs)
            if cand:
                best_candidate = cand

    def exhaustive_sign_search(exps):
        """For fixed exponents, try all 2048 sign patterns (coeffs = ±1)."""
        n = len(exps)
        if n != 12:
            return None
        max_deg = max(exps)
        p_base = np.zeros(max_deg + 1, dtype=np.int64)
        for e in exps:
            p_base[e] = 1  # magnitude 1

        best_local = float('inf')
        best_coeffs = None

        # Precompute: for each sign pattern, the square
        # Use vectorized approach: for each of 2048 patterns
        for mask in range(2048):
            if time.time() > deadline - 2:
                break
            coeffs = []
            for j in range(12):
                coeffs.append(1 if (mask >> j) & 1 else -1)

            p = np.zeros(max_deg + 1, dtype=np.int64)
            for e, c in zip(exps, coeffs):
                p[e] = c
            q = np.convolve(p, p)
            count = int(np.count_nonzero(q))
            if count < best_local:
                best_local = count
                best_coeffs = coeffs[:]

        if best_coeffs is not None:
            return best_local, exps[:], best_coeffs
        return None

    def exhaustive_sign_search_magnitudes(exps, mags):
        """For fixed exponents, try all 2048 sign patterns with given magnitudes."""
        n = len(exps)
        if n != 12:
            return None
        max_deg = max(exps)

        best_local = float('inf')
        best_coeffs = None

        for mask in range(2048):
            if time.time() > deadline - 2:
                break
            coeffs = []
            for j in range(12):
                sign = 1 if (mask >> j) & 1 else -1
                coeffs.append(sign * mags[j])

            p = np.zeros(max_deg + 1, dtype=np.int64)
            for e, c in zip(exps, coeffs):
                p[e] = c
            q = np.convolve(p, p)
            count = int(np.count_nonzero(q))
            if count < best_local:
                best_local = count
                best_coeffs = coeffs[:]

        if best_coeffs is not None:
            return best_local, exps[:], best_coeffs
        return None

    def generate_candidate_exps():
        """Generate diverse candidate exponent sets with good collision structure."""
        candidates = []

        # 1. Arithmetic progressions
        for a in range(0, 10):
            for d in range(1, 5):
                exps = [a + i * d for i in range(12)]
                if max(exps) <= 40 and len(set(exps)) == 12:
                    candidates.append(tuple(exps))

        # 2. Two blocks
        for k in range(2, 11):
            for a in range(0, 15):
                block1 = [a + i for i in range(k)]
                for b in range(a + k, 41):
                    block2 = [b + i for i in range(12 - k)]
                    if max(block2) <= 40:
                        exps = tuple(sorted(block1 + block2))
                        if len(set(exps)) == 12:
                            candidates.append(exps)

        # 3. Symmetric around center
        random.seed(123)
        for _ in range(80):
            left = sorted(random.sample(range(0, 21), 6))
            right = sorted(40 - x for x in left)
            exps = tuple(sorted(left + right))
            if len(set(exps)) == 12:
                candidates.append(exps)

        # 4. Random sets
        random.seed(456)
        for _ in range(100):
            exps = tuple(sorted(random.sample(range(0, 41), 12)))
            candidates.append(exps)

        # 5. Three blocks
        for k1 in range(2, 6):
            for k2 in range(2, 6):
                k3 = 12 - k1 - k2
                if k3 < 2:
                    continue
                for a in range(0, 10):
                    block1 = [a + i for i in range(k1)]
                    for b in range(a + k1 + 1, 30):
                        block2 = [b + i for i in range(k2)]
                        for c in range(b + k2 + 1, 41):
                            block3 = [c + i for i in range(k3)]
                            if max(block3) <= 40:
                                exps = tuple(sorted(block1 + block2 + block3))
                                if len(set(exps)) == 12:
                                    candidates.append(exps)

        # 6. Near-arithmetic with one perturbation
        for a in range(0, 10):
            for d in range(1, 5):
                base = [a + i * d for i in range(12)]
                if max(base) > 40:
                    continue
                for i in range(12):
                    for delta in [-2, -1, 1, 2]:
                        exps_list = base[:]
                        exps_list[i] += delta
                        if 0 <= exps_list[i] <= 40 and len(set(exps_list)) == 12:
                            candidates.append(tuple(sorted(exps_list)))

        # Deduplicate
        seen = set()
        unique = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(list(c))

        return unique

    def simulate_annealing(initial_exps, initial_coeffs, seed, time_limit):
        random.seed(seed)
        exps = list(initial_exps)
        coeffs = list(initial_coeffs)
        current_score = compute_square_nonzeros(exps, coeffs)

        T = 3.0
        cooling = 0.99998
        min_T = 0.005
        coeff_pool = [-1, 1, -2, 2, -3, 3]

        steps = 0
        stagnant_steps = 0
        local_best_score = current_score
        local_best_exps = list(exps)
        local_best_coeffs = list(coeffs)

        while time.time() < time_limit and T > min_T:
            steps += 1
            move_type = random.random()

            if move_type < 0.35:
                i = random.randint(0, 11)
                old_c = coeffs[i]
                new_c = random.choice(coeff_pool)
                if new_c == old_c:
                    continue
                coeffs[i] = new_c
                new_score = compute_square_nonzeros(exps, coeffs)
                delta = new_score - current_score
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_score = new_score
                    if current_score < local_best_score:
                        local_best_score = current_score
                        local_best_exps = list(exps)
                        local_best_coeffs = list(coeffs)
                        stagnant_steps = 0
                    else:
                        stagnant_steps += 1
                else:
                    coeffs[i] = old_c
                    stagnant_steps += 1

            elif move_type < 0.75:
                i = random.randint(0, 11)
                old_exp = exps[i]
                shift = random.choice([-3, -2, -1, 1, 2, 3])
                new_exp = old_exp + shift
                if new_exp < 0 or new_exp > 40:
                    continue
                if new_exp in exps:
                    if exps.index(new_exp) != i:
                        continue
                exps[i] = new_exp
                new_score = compute_square_nonzeros(exps, coeffs)
                delta = new_score - current_score
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_score = new_score
                    if current_score < local_best_score:
                        local_best_score = current_score
                        local_best_exps = list(exps)
                        local_best_coeffs = list(coeffs)
                        stagnant_steps = 0
                    else:
                        stagnant_steps += 1
                else:
                    exps[i] = old_exp
                    stagnant_steps += 1

            else:
                i, j = random.sample(range(12), 2)
                exps[i], exps[j] = exps[j], exps[i]
                new_score = compute_square_nonzeros(exps, coeffs)
                delta = new_score - current_score
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_score = new_score
                    if current_score < local_best_score:
                        local_best_score = current_score
                        local_best_exps = list(exps)
                        local_best_coeffs = list(coeffs)
                        stagnant_steps = 0
                    else:
                        stagnant_steps += 1
                else:
                    exps[i], exps[j] = exps[j], exps[i]
                    stagnant_steps += 1

            T *= cooling

            if stagnant_steps > 500:
                exps = list(local_best_exps)
                coeffs = list(local_best_coeffs)
                current_score = local_best_score
                stagnant_steps = 0
                T = max(T, 1.0)

            if steps % 200 == 0:
                if len(set(exps)) == 12 and 0 <= min(exps) and max(exps) <= 40:
                    if current_score < best_score:
                        paired = sorted(zip(exps, coeffs))
                        e_sorted = [p[0] for p in paired]
                        c_sorted = [p[1] for p in paired]
                        try_update(e_sorted, c_sorted)

        if len(set(exps)) == 12 and 0 <= min(exps) and max(exps) <= 40:
            paired = sorted(zip(exps, coeffs))
            e_sorted = [p[0] for p in paired]
            c_sorted = [p[1] for p in paired]
            try_update(e_sorted, c_sorted)

    # --- Phase 1: Exhaustive sign search over structured exponent sets ---
    candidate_exps = generate_candidate_exps()

    # Sort by a heuristic: prefer sets with more additive collisions
    def collision_count(exps):
        """Count number of pairs (i,j,k,l) with i<j, k<l, i!=k, e_i+e_j=e_k+e_l."""
        from collections import Counter
        sums = Counter()
        for i in range(12):
            for j in range(i + 1, 12):
                sums[exps[i] + exps[j]] += 1
        collisions = 0
        for s, cnt in sums.items():
            if cnt > 1:
                collisions += cnt * (cnt - 1) // 2
        return collisions

    # Sort by collision count descending (more collisions = more cancellation potential)
    candidate_exps.sort(key=lambda e: collision_count(e), reverse=True)

    # Evaluate top candidates with exhaustive sign search
    num_sign_searches = 0
    for exps in candidate_exps:
        if time.time() > deadline - 10:
            break
        result = exhaustive_sign_search(exps)
        num_sign_searches += 1
        if result is not None:
            score, e, c = result
            if score < best_score:
                try_update(e, c)
                if best_score <= 12:
                    break

    # --- Phase 2: Try some with non-unit magnitudes ---
    if time.time() < deadline - 15:
        # Focus on top-collision exponent sets
        top_exps = candidate_exps[:20]
        magnitude_patterns = [
            [1]*12,
            [1,1,1,1,1,1,2,2,2,2,2,2],
            [1,1,1,1,2,2,2,2,1,1,1,1],
            [1,2,1,2,1,2,1,2,1,2,1,2],
            [2,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,2,2,2,2,2,2,2],
        ]
        for exps in top_exps:
            if time.time() > deadline - 5:
                break
            for mag in magnitude_patterns:
                if time.time() > deadline - 3:
                    break
                result = exhaustive_sign_search_magnitudes(exps, mag)
                if result is not None:
                    score, e, c = result
                    if score < best_score:
                        try_update(e, c)
                        if best_score <= 12:
                            break

    # --- Phase 3: Simulated Annealing on best candidates ---
    if time.time() < deadline - 5:
        # Start SA from the best candidate found so far
        if best_score < 21:
            best_exps = [i for i, c in enumerate(best_candidate) if c != 0]
            best_coeffs = [best_candidate[i] for i in best_exps]
            simulate_annealing(best_exps, best_coeffs, seed=99, time_limit=time.time() + 8)
        else:
            # Start from a few good structured points
            for exps in candidate_exps[:5]:
                if time.time() > deadline - 3:
                    break
                coeffs = [1 if i % 2 == 0 else -1 for i in range(12)]
                simulate_annealing(exps, coeffs, seed=100, time_limit=time.time() + 5)

    # --- Phase 4: Final polish ---
    if best_candidate and best_score < 21 and time.time() < deadline - 2:
        best_exps = [i for i, c in enumerate(best_candidate) if c != 0]
        best_coeffs = [best_candidate[i] for i in best_exps]

        for i in range(12):
            old_c = best_coeffs[i]
            for new_c in [-3, -2, -1, 1, 2, 3]:
                if new_c == 0 or new_c == old_c:
                    continue
                best_coeffs[i] = new_c
                s = compute_square_nonzeros(best_exps, best_coeffs)
                if s < best_score:
                    best_score = s
                else:
                    best_coeffs[i] = old_c

            old_e = best_exps[i]
            for delta in [-2, -1, 1, 2]:
                new_e = old_e + delta
                if new_e < 0 or new_e > 40:
                    continue
                if new_e in best_exps:
                    if best_exps.index(new_e) != i:
                        continue
                best_exps[i] = new_e
                paired = sorted(zip(best_exps, best_coeffs))
                best_exps = [p[0] for p in paired]
                best_coeffs = [p[1] for p in paired]
                s = compute_square_nonzeros(best_exps, best_coeffs)
                if s < best_score:
                    best_score = s
                else:
                    paired = sorted(zip([e if e != new_e else old_e for e in best_exps], best_coeffs))
                    best_exps = [p[0] for p in paired]
                    best_coeffs = [p[1] for p in paired]

        if best_score < 21:
            cand = get_dense(best_exps, best_coeffs)
            if cand:
                best_candidate = cand

    return best_candidate

# EVOLVE-BLOCK-END
