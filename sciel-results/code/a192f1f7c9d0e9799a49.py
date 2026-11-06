"""Sparse squares of twelve-term integer polynomials - graph-theoretic structure search."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = start + 85.0

    best_poly = None
    best_nz = float('inf')
    best_exps = None
    best_coefs = None

    def square_nonzero_fast(exponents, coeffs):
        """Compute the number of nonzero terms in the square using sparse convolution."""
        c = np.asarray(coeffs, dtype=np.int64)
        e = np.asarray(exponents, dtype=np.int64)
        # Build the full coefficient array
        max_deg = max(e)
        size = 2 * max_deg + 1
        poly = np.zeros(size, dtype=np.int64)
        for i in range(12):
            poly[e[i]] = c[i]
        # Square via convolution
        result = np.convolve(poly, poly)
        return int(np.count_nonzero(result))

    def square_coefficients(exponents, coeffs):
        """Return the full coefficient array of the square."""
        c = np.asarray(coeffs, dtype=np.int64)
        e = np.asarray(exponents, dtype=np.int64)
        max_deg = max(e)
        size = 2 * max_deg + 1
        poly = np.zeros(size, dtype=np.int64)
        for i in range(12):
            poly[e[i]] = c[i]
        return np.convolve(poly, poly)

    def count_sum_collisions(exponents):
        """Count how many distinct sums e_i+e_j (i<j) collide (appear more than once)."""
        e = np.asarray(exponents, dtype=np.int64)
        iu = np.triu_indices(12, k=1)
        pair_sums = e[iu[0]] + e[iu[1]]
        unique, counts = np.unique(pair_sums, return_counts=True)
        # Number of pairs involved in collisions
        collided_pairs = int(np.sum(counts[counts > 1]))
        return collided_pairs

    def validate(poly_coeff_list):
        if not poly_coeff_list or len(poly_coeff_list) > 41:
            return False
        if poly_coeff_list[-1] == 0:
            return False
        for c in poly_coeff_list:
            if not isinstance(c, int) or c < -64 or c > 64:
                return False
        if sum(1 for c in poly_coeff_list if c != 0) != 12:
            return False
        return True

    def update_best(exponents, coeffs, nz):
        nonlocal best_poly, best_nz, best_exps, best_coefs
        if nz < best_nz:
            best_nz = nz
            best_exps = list(exponents)
            best_coefs = list(coeffs)
            max_deg = max(exponents)
            poly = [0] * (max_deg + 1)
            for e, c in zip(exponents, coeffs):
                poly[e] = c
            best_poly = poly

    def solve_coefficients_for_min_nz(exponents, max_trials=200):
        """Given fixed exponents, search over coefficient assignments to minimize square nz."""
        e = np.asarray(exponents, dtype=np.int64)
        best_local_nz = float('inf')
        best_local_coefs = None

        # Precompute all pair sums
        iu = np.triu_indices(12, k=1)
        pair_sums = e[iu[0]] + e[iu[1]]
        pair_indices = (iu[0], iu[1])

        # Try structured coefficient patterns first
        patterns = []
        # All ones
        patterns.append([1]*12)
        # Alternating
        patterns.append([1 if i%2==0 else -1 for i in range(12)])
        # Small values
        for _ in range(max_trials):
            if random.random() < 0.3:
                coefs = [random.choice([-1, 1]) for _ in range(12)]
            elif random.random() < 0.5:
                coefs = [random.choice([-2, -1, 1, 2]) for _ in range(12)]
            else:
                coefs = [random.choice([-4, -3, -2, -1, 1, 2, 3, 4]) for _ in range(12)]
            patterns.append(coefs)

        for coefs in patterns:
            if any(c == 0 or abs(c) > 64 for c in coefs):
                continue
            nz = square_nonzero_fast(exponents, coefs)
            if nz < best_local_nz:
                best_local_nz = nz
                best_local_coefs = list(coefs)

        # Local search on coefficients
        if best_local_coefs is not None:
            cur_coefs = list(best_local_coefs)
            cur_nz = best_local_nz
            for _ in range(500):
                if time.time() > deadline - 5:
                    break
                idx = random.randint(0, 11)
                new_val = random.choice([-8,-6,-4,-3,-2,-1,1,2,3,4,5,6,8,12,16])
                if new_val == cur_coefs[idx]:
                    continue
                old_val = cur_coefs[idx]
                cur_coefs[idx] = new_val
                new_nz = square_nonzero_fast(exponents, cur_coefs)
                if new_nz <= cur_nz:
                    cur_nz = new_nz
                    if cur_nz < best_local_nz:
                        best_local_nz = cur_nz
                        best_local_coefs = list(cur_coefs)
                else:
                    cur_coefs[idx] = old_val

        return best_local_nz, best_local_coefs

    def simulated_annealing(init_exps, init_coefs, max_iters=3000):
        """SA over both exponents and coefficients."""
        nonlocal best_poly, best_nz, best_exps, best_coefs
        cur_exps = sorted(init_exps)
        cur_coefs = list(init_coefs)
        cur_nz = square_nonzero_fast(cur_exps, cur_coefs)
        update_best(cur_exps, cur_coefs, cur_nz)

        T = 8.0
        T_min = 0.01
        cooling = 0.9997

        for it in range(max_iters):
            if time.time() > deadline - 3:
                break

            new_exps = list(cur_exps)
            new_coefs = list(cur_coefs)
            move_type = random.random()

            if move_type < 0.25:
                # Single exponent move
                idx = random.randint(0, 11)
                shift = random.randint(-6, 6)
                new_e = new_exps[idx] + shift
                if 0 <= new_e < 41 and new_e not in new_exps:
                    new_exps[idx] = new_e
                    new_exps.sort()
                else:
                    continue
            elif move_type < 0.40:
                # Two exponent move
                i, j = random.sample(range(12), 2)
                si = random.randint(-4, 4)
                sj = random.randint(-4, 4)
                ne_i = new_exps[i] + si
                ne_j = new_exps[j] + sj
                if (0 <= ne_i < 41 and 0 <= ne_j < 41 and
                    ne_i not in new_exps and ne_j not in new_exps and ne_i != ne_j):
                    new_exps[i] = ne_i
                    new_exps[j] = ne_j
                    new_exps.sort()
                else:
                    continue
            elif move_type < 0.50:
                # Collision-guided: try to create a sum collision
                # Pick two disjoint pairs and try to align their sums
                i, j = random.sample(range(12), 2)
                remaining = [x for x in range(12) if x not in (i, j)]
                if len(remaining) < 2:
                    continue
                k, l = random.sample(remaining, 2)
                target_sum = new_exps[i] + new_exps[j]
                # Try to move k so that new_exps[k] + new_exps[l] == target_sum
                new_k = target_sum - new_exps[l]
                if 0 <= new_k < 41 and new_k not in new_exps:
                    new_exps[k] = new_k
                    new_exps.sort()
                else:
                    # Try moving l
                    new_l = target_sum - new_exps[k]
                    if 0 <= new_l < 41 and new_l not in new_exps:
                        new_exps[l] = new_l
                        new_exps.sort()
                    else:
                        continue
            elif move_type < 0.65:
                # Coefficient modification
                idx = random.randint(0, 11)
                new_val = random.choice([-8,-6,-4,-3,-2,-1,1,2,3,4,5,6,8,12,16])
                if new_val != new_coefs[idx] and abs(new_val) <= 64:
                    new_coefs[idx] = new_val
                else:
                    continue
            elif move_type < 0.75:
                # Two coefficient modification
                i, j = random.sample(range(12), 2)
                new_coefs[i] = random.choice([-4,-3,-2,-1,1,2,3,4,5,6])
                new_coefs[j] = random.choice([-4,-3,-2,-1,1,2,3,4,5,6])
            else:
                # Swap two exponents with nearby values
                i, j = random.sample(range(12), 2)
                if new_exps[i] != new_exps[j]:
                    new_exps[i], new_exps[j] = new_exps[j], new_exps[i]
                    new_exps.sort()

            if len(set(new_exps)) != 12 or any(e < 0 or e >= 41 for e in new_exps):
                continue
            if any(c == 0 or abs(c) > 64 for c in new_coefs):
                continue

            new_nz = square_nonzero_fast(new_exps, new_coefs)
            delta = new_nz - cur_nz

            if delta <= 0 or random.random() < np.exp(-delta / max(T, T_min)):
                cur_nz = new_nz
                cur_exps = list(new_exps)
                cur_coefs = list(new_coefs)
                update_best(cur_exps, cur_coefs, cur_nz)

            T = max(T * cooling, T_min)

        return cur_exps, cur_coefs, cur_nz

    random.seed(42)

    # Phase 1: Generate structured seeds with high collision density
    seeds = []

    # Arithmetic progressions (highly structured, many sum collisions)
    for d in range(1, 7):
        max_a = max(0, 40 - 11 * d)
        for a in range(0, max_a + 1):
            exps = [a + i * d for i in range(12)]
            if exps[-1] >= 41:
                continue
            for coefs in [[1]*12, [1 if i%2==0 else -1 for i in range(12)]]:
                seeds.append((exps, coefs))

    # Two interleaved APs
    for d1 in range(1, 4):
        for d2 in range(1, 4):
            for a1 in range(0, 12):
                for a2 in range(0, 12):
                    exps1 = [a1 + i*d1 for i in range(6)]
                    exps2 = [a2 + i*d2 for i in range(6)]
                    exps = sorted(exps1 + exps2)
                    if len(set(exps)) != 12 or any(e >= 41 for e in exps):
                        continue
                    for s in [1, -1]:
                        coefs = [s]*6 + [s]*6
                        seeds.append((exps, coefs))

    # Symmetric structures around a center
    for center in range(6, 35):
        for offsets in [[1,2,3,4,5,6],[1,2,3,4,5,8],[1,2,4,5,7,9],[1,3,4,5,6,8],[2,3,4,5,6,7]]:
            exps = sorted([center-o for o in offsets] + [center+o for o in offsets])
            if any(e < 0 or e >= 41 for e in exps) or len(set(exps)) != 12:
                continue
            for coefs in [[1]*12, [1]*6+[-1]*6, [1,-1,1,-1,1,-1]*2]:
                seeds.append((exps, coefs))

    # Cubic-like structures
    for a in range(0, 8):
        for b in range(1, 5):
            exps = sorted([a + b*i + i*i for i in range(12)])
            if any(e >= 41 for e in exps) or len(set(exps)) != 12:
                continue
            seeds.append((exps, [1]*12))

    # Evaluate seeds
    for exps, coefs in seeds:
        if time.time() > deadline - 40:
            break
        if len(set(exps)) != 12 or any(e < 0 or e >= 41 for e in exps):
            continue
        if any(c == 0 or abs(c) > 64 for c in coefs):
            continue
        nz = square_nonzero_fast(exps, coefs)
        update_best(exps, coefs, nz)

    # Phase 2: For top seed structures, optimize coefficients
    if best_exps is not None and time.time() < deadline - 25:
        # Re-evaluate best with coefficient optimization
        nz_opt, coefs_opt = solve_coefficients_for_min_nz(best_exps, max_trials=100)
        if coefs_opt is not None:
            update_best(best_exps, coefs_opt, nz_opt)

    # Phase 3: SA from multiple diverse starting points
    sa_starts = []

    # Start from best
    if best_exps is not None:
        sa_starts.append((best_exps, best_coefs))
        # Perturbed versions
        for _ in range(3):
            exps = list(best_exps)
            coefs = list(best_coefs)
            for _ in range(random.randint(2, 4)):
                if random.random() < 0.5:
                    idx = random.randint(0, 11)
                    shift = random.randint(-4, 4)
                    new_e = exps[idx] + shift
                    if 0 <= new_e < 41 and new_e not in exps:
                        exps[idx] = new_e
                        exps.sort()
                else:
                    idx = random.randint(0, 11)
                    coefs[idx] = random.choice([-4,-3,-2,-1,1,2,3,4,5,6])
                    if coefs[idx] == 0:
                        coefs[idx] = 1
            if len(set(exps)) == 12 and all(0 <= e < 41 for e in exps) and \
               all(c != 0 and abs(c) <= 64 for c in coefs):
                sa_starts.append((exps, coefs))

    # Random structured starts
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        rtype = random.random()
        if rtype < 0.3:
            d = random.randint(1, 5)
            max_a = max(0, 40 - 11 * d)
            a = random.randint(0, max_a)
            exps = [a + i * d for i in range(12)]
            if exps[-1] >= 41:
                continue
        elif rtype < 0.6:
            d1 = random.randint(1, 3)
            d2 = random.randint(1, 3)
            a1 = random.randint(0, 8)
            a2 = random.randint(0, 8)
            exps = sorted([a1 + i*d1 for i in range(6)] + [a2 + i*d2 for i in range(6)])
            if len(set(exps)) != 12 or any(e >= 41 for e in exps):
                continue
        else:
            exps = sorted(random.sample(range(41), 12))
        coefs = [random.choice([-3,-2,-1,1,2,3]) for _ in range(12)]
        sa_starts.append((exps, coefs))

    # Run SA
    for exps, coefs in sa_starts:
        if time.time() > deadline - 5:
            break
        iters = 3000 if time.time() < deadline - 30 else 1500
        simulated_annealing(exps, coefs, max_iters=iters)

    # Phase 4: Intensive local search on best
    if best_exps is not None and time.time() < deadline - 5:
        # Coefficient polish
        nz_opt, coefs_opt = solve_coefficients_for_min_nz(best_exps, max_trials=300)
        if coefs_opt is not None:
            update_best(best_exps, coefs_opt, nz_opt)

        # Final SA burst
        if time.time() < deadline - 3:
            simulated_annealing(best_exps, best_coefs, max_iters=2000)

    # Final fallback
    if best_poly is None:
        best_poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    if not validate(best_poly):
        best_poly = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    return best_poly

# EVOLVE-BLOCK-END
