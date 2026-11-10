import numpy as np
import time
import random
from itertools import product, combinations

def construct():
    start_time = time.time()
    deadline = start_time + 83  # Safety margin before 90s kill

    def count_nonzero(poly):
        return int(np.count_nonzero(poly))

    def poly_square(coeffs):
        return np.convolve(coeffs, coeffs)

    def evaluate(coeffs):
        sq = poly_square(coeffs)
        return count_nonzero(sq)

    def evaluate_fast(coeffs):
        """Compute square coefficients and count nonzeros, using precomputed pairs."""
        total = 23
        sq = [0] * total
        for k in range(total):
            s = 0
            for i, j in pair_map[k]:
                s += coeffs[i] * coeffs[j]
            sq[k] = s
        nz = sum(1 for v in sq if v != 0)
        return nz, sq

    best_score = float('inf')
    best_poly = [1] * 12  # baseline

    def try_poly(coeffs):
        nonlocal best_score, best_poly
        if len(coeffs) != 12:
            return
        if sum(1 for c in coeffs if c != 0) != 12:
            return
        if any(c < -64 or c > 64 for c in coeffs):
            return
        score = evaluate(coeffs)
        if score < best_score:
            best_score = score
            best_poly = list(coeffs)

    # Precompute: for each k in 0..22, the pairs (i,j) with i+j=k, 0<=i,j<=11
    def get_pairs(k):
        pairs = []
        for i in range(max(0, k-11), min(11, k) + 1):
            j = k - i
            if 0 <= j <= 11:
                pairs.append((i, j))
        return pairs

    pair_map = {k: get_pairs(k) for k in range(23)}

    # Try baseline
    try_poly([1] * 12)

    # === CONSTRAINT SATISFACTION APPROACH ===
    # For P(x) = sum_{i=0}^{11} a_i x^i, P(x)^2 = sum_{k=0}^{22} c_k x^k
    # where c_k = sum_{i+j=k} a_i * a_j
    # We want to maximize the number of k where c_k = 0.
    #
    # Key insight: if we fix the support (which a_i are nonzero) and the
    # "zero pattern" (which c_k must be zero), we get a system of quadratic
    # equations. We can use backtracking to find integer solutions.
    #
    # Strategy: enumerate candidate zero-patterns for the square coefficients,
    # then solve the resulting system via constrained backtracking.

    # Precompute: for each k in 0..22, the pairs (i,j) with i+j=k, 0<=i,j<=11
    def get_pairs(k):
        pairs = []
        for i in range(max(0, k-11), min(11, k) + 1):
            j = k - i
            if 0 <= j <= 11:
                pairs.append((i, j))
        return pairs

    pair_map = {k: get_pairs(k) for k in range(23)}

    # For a given target zero-set Z (subset of {0,...,22}), check if there
    # exists a 12-term polynomial whose square is zero on Z.
    # Use randomized local search on the coefficient vector, with the
    # additional constraint that c_k = 0 for all k in Z.

    def evaluate_with_zeros(coeffs, zero_set):
        """Return (num_nonzero_square, num_zero_violations) using fast eval."""
        nz, sq = evaluate_fast(coeffs)
        violations = sum(1 for k in zero_set if sq[k] != 0)
        return nz, violations

    def solve_with_zero_constraint(zero_set, max_iters=5000, seed=42):
        """Local search to find a 12-term poly whose square vanishes on zero_set."""
        rng = random.Random(seed)
        best_viol = float('inf')
        best_nz = float('inf')
        best_coeffs = None

        # Start with random coefficients
        coeffs = [rng.choice([-1, 1]) * rng.randint(1, 8) for _ in range(12)]

        curr_nz, curr_viol = evaluate_with_zeros(coeffs, zero_set)
        best_viol = curr_viol
        best_nz = curr_nz
        best_coeffs = list(coeffs)

        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Try modifying one coefficient
            pos = rng.randint(0, 11)
            old_val = coeffs[pos]

            # Generate candidate values biased toward creating cancellations
            candidates = set()
            for other in range(12):
                if other != pos:
                    v = coeffs[other]
                    for mult in [-4, -3, -2, -1, 1, 2, 3, 4]:
                        cv = v * mult
                        if -64 <= cv <= 64 and cv != old_val:
                            candidates.add(cv)
                    # Also try to cancel specific square terms
                    for k in range(23):
                        for i2, j2 in pair_map[k]:
                            if i2 == pos and j2 != pos:
                                # Need coeffs[pos] * coeffs[j2] to cancel with other terms
                                # Try values that make partial sums zero
                                partial = sum(coeffs[i3] * coeffs[j3] for i3, j3 in pair_map[k]
                                              if not (i3 == pos and j3 == j2) and not (i3 == j2 and j3 == pos))
                                if coeffs[j2] != 0 and partial != 0:
                                    target = -partial // coeffs[j2]
                                    if -64 <= target <= 64 and target != old_val:
                                        candidates.add(target)
                                    # Also try nearby values
                                    for delta in [-2, -1, 1, 2]:
                                        t2 = target + delta
                                        if -64 <= t2 <= 64 and t2 != old_val:
                                            candidates.add(t2)
                            elif j2 == pos and i2 != pos:
                                partial = sum(coeffs[i3] * coeffs[j3] for i3, j3 in pair_map[k]
                                              if not (i3 == pos and j3 == i2) and not (i3 == i2 and j3 == pos))
                                if coeffs[i2] != 0 and partial != 0:
                                    target = -partial // coeffs[i2]
                                    if -64 <= target <= 64 and target != old_val:
                                        candidates.add(target)
                                    for delta in [-2, -1, 1, 2]:
                                        t2 = target + delta
                                        if -64 <= t2 <= 64 and t2 != old_val:
                                            candidates.add(t2)

            for _ in range(4):
                cv = rng.randint(-20, 20)
                if cv != old_val:
                    candidates.add(cv)

            candidates = [v for v in candidates if -64 <= v <= 64 and v != old_val and v != 0]
            if not candidates:
                # Allow zero temporarily
                candidates = [v for v in candidates if -64 <= v <= 64 and v != old_val]
                if not candidates:
                    continue

            new_val = rng.choice(candidates)
            coeffs[pos] = new_val

            # Ensure exactly 12 nonzero (if we set to 0, we need to fix)
            nz_count = sum(1 for c in coeffs if c != 0)
            if nz_count != 12:
                if nz_count < 12:
                    zero_positions = [i for i in range(12) if coeffs[i] == 0]
                    if zero_positions:
                        zp = rng.choice(zero_positions)
                        coeffs[zp] = rng.choice([-1, 1]) * rng.randint(1, 4)
                        nz_count = sum(1 for c in coeffs if c != 0)
                else:
                    pass

            if nz_count != 12:
                coeffs[pos] = old_val
                continue

            new_nz, new_viol = evaluate_with_zeros(coeffs, zero_set)
            # Objective: minimize violations first, then nonzero count
            if (new_viol, new_nz) < (curr_viol, curr_nz):
                curr_viol = new_viol
                curr_nz = new_nz
                if (new_viol, new_nz) < (best_viol, best_nz):
                    best_viol = new_viol
                    best_nz = new_nz
                    best_coeffs = list(coeffs)
            else:
                # With some probability, accept worse to escape local minima
                if rng.random() < 0.08 and new_viol <= curr_viol + 1:
                    curr_viol = new_viol
                    curr_nz = new_nz
                else:
                    coeffs[pos] = old_val

        return best_coeffs, best_viol, best_nz

    # Enumerate candidate zero-sets from most constrained to least
    # The square has 23 coefficients. We want to find which subsets can be zero.
    # Start with small zero-sets (easy to satisfy) and grow.

    # Phase 1: Quick structured search to get a good baseline
    for restart in range(300):
        if time.time() > deadline:
            break
        # Random structured polynomial
        pattern = restart % 8
        if pattern == 0:
            poly = [random.choice([-1, 1]) * random.randint(1, 4) for _ in range(12)]
        elif pattern == 1:
            # Palindromic
            half = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(6)]
            poly = half + half[::-1]
            for i in range(12):
                if poly[i] == 0:
                    poly[i] = 1
        elif pattern == 2:
            # Alternating
            poly = [random.choice([-1, 1]) * (1 + i % 2) * random.randint(1, 3) for i in range(12)]
            for i in range(12):
                if poly[i] == 0:
                    poly[i] = 1
        elif pattern == 3:
            # Two-block
            k = random.randint(3, 9)
            s = random.choice([-1, 1])
            poly = [s * random.randint(1, 3) for _ in range(k)] + [-s * random.randint(1, 3) for _ in range(12-k)]
        elif pattern == 4:
            # Sparse magnitudes
            poly = [random.choice([-4, -2, -1, 1, 2, 4]) for _ in range(12)]
        elif pattern == 5:
            # Arithmetic progression of signs
            base = random.choice([-2, -1, 1, 2])
            poly = [base * ((i % 3) + 1) * random.choice([-1, 1]) for i in range(12)]
            for i in range(12):
                if poly[i] == 0:
                    poly[i] = 1
        elif pattern == 6:
            # Fibonacci-like
            poly = [1, 1]
            for i in range(10):
                poly.append(poly[-1] + poly[-2])
            poly = [p * random.choice([-1, 1]) for p in poly[:12]]
            for i in range(12):
                if poly[i] == 0 or abs(poly[i]) > 64:
                    poly[i] = random.choice([-1, 1]) * random.randint(1, 4)
        else:
            # Geometric-like with sign flips
            base = random.choice([2, 3])
            poly = [base ** (i % 4) * random.choice([-1, 1]) for i in range(12)]
            for i in range(12):
                if abs(poly[i]) > 64:
                    poly[i] = random.choice([-1, 1]) * random.randint(1, 4)

        try_poly(poly)

    # Phase 2: Constraint-based search
    # Try to find polynomials with specific zero patterns in their square
    # Focus on patterns that are "natural": e.g., c_0=0 (impossible since a_0^2),
    # so c_0 is always nonzero. c_22 = a_11^2, also always nonzero.
    # So the interior coefficients c_1..c_21 are the candidates for zero.

    # Try various zero-set sizes and patterns
    zero_set_candidates = []

    # Small zero sets (1-3 zeros)
    for size in [1, 2, 3]:
        for k in range(1, 22):  # c_0 and c_22 can't be zero
            if size == 1:
                zero_set_candidates.append(frozenset([k]))
            elif size == 2:
                for j in range(k+1, 22):
                    zero_set_candidates.append(frozenset([k, j]))
            else:
                if len(zero_set_candidates) > 500:
                    break
                for j in range(k+1, 22):
                    for l in range(j+1, 22):
                        zero_set_candidates.append(frozenset([k, j, l]))
                        if len(zero_set_candidates) > 500:
                            break
                    if len(zero_set_candidates) > 500:
                        break
        if len(zero_set_candidates) > 500:
            break

    # Also try structured zero patterns: e.g., all even indices, all odd, etc.
    structured_zero_sets = [
        frozenset(range(1, 22, 2)),  # odd indices
        frozenset(range(2, 22, 2)),  # even indices (excluding 0,22)
        frozenset(range(1, 11)),     # first half
        frozenset(range(12, 22)),    # second half
        frozenset(range(2, 21, 3)),  # every 3rd
        frozenset(range(1, 22, 4)),  # every 4th
    ]
    zero_set_candidates.extend(structured_zero_sets)

    random.shuffle(zero_set_candidates)

    # Solve for each zero set
    for zs in zero_set_candidates:
        if time.time() > deadline:
            break

        # Skip if we already have a score <= len(zs) complement
        # (i.e., we already found a poly with at least as many zeros)
        target_nz = 23 - len(zs)
        if best_score <= target_nz:
            continue

        # Try multiple seeds for diversity
        for seed in [42, 123, 777]:
            if time.time() > deadline:
                break
            coeffs, viol, nz = solve_with_zero_constraint(zs, max_iters=3000, seed=seed)
            if coeffs is not None:
                # Verify and update
                actual_nz = evaluate(coeffs)
                if actual_nz < best_score:
                    best_score = actual_nz
                    best_poly = list(coeffs)

                # Also try local refinement from this solution
                if time.time() < deadline:
                    for _ in range(500):
                        if time.time() > deadline:
                            break
                        pos = random.randint(0, 11)
                        old_val = best_poly[pos]
                        new_val = random.choice([-3, -2, -1, 1, 2, 3,
                                                 best_poly[(pos+1)%12],
                                                 -best_poly[(pos+1)%12],
                                                 best_poly[(pos+2)%12],
                                                 -best_poly[(pos+2)%12]])
                        if new_val == old_val or new_val == 0:
                            continue
                        if not (-64 <= new_val <= 64):
                            continue
                        best_poly[pos] = new_val
                        new_score = evaluate(best_poly)
                        if new_score < best_score:
                            best_score = new_score
                        else:
                            best_poly[pos] = old_val

    # Phase 3: Final local refinement with targeted candidates
    if time.time() < deadline:
        for _round in range(3):
            if time.time() > deadline - 5:
                break
            for pos in range(12):
                if time.time() > deadline:
                    break
                old_val = best_poly[pos]
                # Generate smart candidates: values related to other coefficients
                candidates = set()
                for other in range(12):
                    if other != pos:
                        v = best_poly[other]
                        for mult in [-4, -3, -2, -1, 1, 2, 3, 4]:
                            cv = v * mult
                            if -64 <= cv <= 64 and cv != old_val and cv != 0:
                                candidates.add(cv)
                # Also try to cancel specific square terms
                for k in range(23):
                    for i2, j2 in pair_map[k]:
                        if i2 == pos and j2 != pos and best_poly[j2] != 0:
                            partial = sum(best_poly[i3] * best_poly[j3] for i3, j3 in pair_map[k]
                                          if not (i3 == pos and j3 == j2) and not (i3 == j2 and j3 == pos))
                            if partial != 0:
                                target = -partial // best_poly[j2]
                                if -64 <= target <= 64 and target != old_val and target != 0:
                                    candidates.add(target)
                                for delta in [-3, -2, -1, 1, 2, 3]:
                                    t2 = target + delta
                                    if -64 <= t2 <= 64 and t2 != old_val and t2 != 0:
                                        candidates.add(t2)
                        elif j2 == pos and i2 != pos and best_poly[i2] != 0:
                            partial = sum(best_poly[i3] * best_poly[j3] for i3, j3 in pair_map[k]
                                          if not (i3 == pos and j3 == i2) and not (i3 == i2 and j3 == pos))
                            if partial != 0:
                                target = -partial // best_poly[i2]
                                if -64 <= target <= 64 and target != old_val and target != 0:
                                    candidates.add(target)
                                for delta in [-3, -2, -1, 1, 2, 3]:
                                    t2 = target + delta
                                    if -64 <= t2 <= 64 and t2 != old_val and t2 != 0:
                                        candidates.add(t2)
                
                # Add some random values
                for _ in range(5):
                    rv = random.randint(-16, 16)
                    if rv != old_val and rv != 0:
                        candidates.add(rv)
                
                # Also try all values in a smaller range
                for nv in range(-12, 13):
                    if nv != old_val and nv != 0:
                        candidates.add(nv)
                
                for new_val in candidates:
                    if time.time() > deadline:
                        break
                    if new_val == old_val or new_val == 0:
                        continue
                    best_poly[pos] = new_val
                    new_score = evaluate(best_poly)
                    if new_score < best_score:
                        best_score = new_score
                    else:
                        best_poly[pos] = old_val
                best_poly[pos] = old_val

    # Phase 4: Exhaustive small-range refinement
    if time.time() < deadline - 3:
        for pos in range(12):
            if time.time() > deadline:
                break
            old_val = best_poly[pos]
            for new_val in range(-20, 21):
                if new_val == old_val or new_val == 0:
                    continue
                best_poly[pos] = new_val
                new_score = evaluate(best_poly)
                if new_score < best_score:
                    best_score = new_score
                else:
                    best_poly[pos] = old_val
            best_poly[pos] = old_val

    # Phase 5: One more targeted pass
    if time.time() < deadline - 1:
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

    return best_poly
