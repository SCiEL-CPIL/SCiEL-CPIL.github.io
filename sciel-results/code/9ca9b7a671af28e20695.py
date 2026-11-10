import numpy as np
import time
import random
from itertools import product, combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill

    def count_nonzero(poly):
        return int(np.count_nonzero(poly))

    def poly_square(coeffs):
        return np.convolve(coeffs, coeffs)

    def evaluate(coeffs):
        sq = poly_square(coeffs)
        return count_nonzero(sq)

    best_score = float('inf')
    best_poly = [1] * 12

    def try_poly(coeffs):
        nonlocal best_score, best_poly
        if len(coeffs) != 12:
            return
        if sum(1 for c in coeffs if c != 0) != 12:
            return
        if any(c < -64 or c > 64 for c in coeffs):
            return
        if coeffs[-1] == 0:
            return
        score = evaluate(coeffs)
        if score < best_score:
            best_score = score
            best_poly = list(coeffs)

    # Baseline
    try_poly([1] * 12)

    # ------------------------------------------------------------------
    # STRUCTURAL CORE: model cancellation as a constraint problem.
    # For p = sum a_i x^i, (p^2)_d = sum_{i+j=d} a_i a_j.
    # We want as many d-buckets to sum to zero as possible.
    # Build the list of "collisions": for each d, the multiset of index
    # pairs (i,j) with i+j=d. A bucket is zero iff the weighted sum cancels.
    # We search assignments (a_0..a_11) from a small structured value set
    # using a backtracking that prunes on partial bucket sums.
    # ------------------------------------------------------------------

    # Precompute collision structure: buckets[d] = list of (i,j) pairs
    buckets = {}
    for i in range(12):
        for j in range(12):
            d = i + j
            buckets.setdefault(d, []).append((i, j))

    # For each degree d, the contributing index-set (unique indices)
    # and the pair list. We'll track which indices are "active" (assigned).
    # Value set: keep small for tractability of backtracking.
    value_sets = [
        [-1, 1],
        [-2, -1, 1, 2],
        [-3, -2, -1, 1, 2, 3],
        [-2, -1, 1, 2, 4, -4],
        [-1, 1, 2, -2, 3, -3],
    ]

    def backtrack_solve(vals, max_nodes=200000):
        """Backtrack over indices 0..11 assigning values, pruning when a
        completed bucket cannot possibly reach zero. Returns best found."""
        n = 12
        assigned = [None] * n
        # partial sum per degree
        deg_sum = [0] * 25
        # count of pairs contributing to each degree whose both indices are assigned
        deg_done = [0] * 25
        # total pairs per degree
        deg_total = [len(buckets.get(d, [])) for d in range(25)]

        nodes = [0]
        best_local = [float('inf'), None]

        def feasible(d):
            # bucket d is "closed" when all its pairs have both endpoints assigned
            if deg_done[d] < deg_total[d]:
                return True  # not yet closed, no constraint
            return deg_sum[d] == 0

        def assign(idx, val):
            assigned[idx] = val
            for j in range(n):
                if j < idx:
                    d = idx + j
                    deg_sum[d] += val * assigned[j]
                    deg_done[d] += 1
            # diagonal
            d = 2 * idx
            deg_sum[d] += val * val
            deg_done[d] += 1

        def unassign(idx, val):
            assigned[idx] = None
            for j in range(n):
                if j < idx:
                    d = idx + j
                    deg_sum[d] -= val * assigned[j]
                    deg_done[d] -= 1
            d = 2 * idx
            deg_sum[d] -= val * val
            deg_done[d] -= 1

        def dfs(idx):
            if nodes[0] > max_nodes or time.time() > deadline:
                return
            nodes[0] += 1
            if idx == n:
                # all assigned; check all buckets
                for d in range(25):
                    if deg_total[d] > 0 and deg_sum[d] != 0:
                        return
                score = evaluate(assigned)
                if score < best_local[0]:
                    best_local[0] = score
                    best_local[1] = list(assigned)
                return
            for v in vals:
                if v == 0:
                    continue
                assign(idx, v)
                # prune: check any bucket that just became closed
                ok = True
                for d in range(idx, 2 * idx + 1):
                    if deg_done[d] == deg_total[d] and deg_total[d] > 0 and deg_sum[d] != 0:
                        ok = False
                        break
                if ok:
                    dfs(idx + 1)
                unassign(idx, v)

        dfs(0)
        return best_local

    # Run backtracking with several value sets
    for vs in value_sets:
        if time.time() > deadline - 5:
            break
        score, poly = backtrack_solve(vs, max_nodes=150000)
        if poly is not None:
            try_poly(poly)

    # ------------------------------------------------------------------
    # STRUCTURED CANDIDATES (from both programs): periodic, geometric,
    # block-sign, cyclotomic-like patterns.
    # ------------------------------------------------------------------
    structured_candidates = [
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, 2, 1, -1, -2, -1, 1, 2, 1, -1, -2, -1],
        [1, -2, 1, 1, -2, 1, 1, -2, 1, 1, -2, 1],
        [1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1],
        [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6],
        [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
        [1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1],
        [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1],
        [1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1],
        [2, -1, 0, 0, 2, -1, 0, 0, 2, -1, 0, 0],  # will be filtered (zeros)
    ]
    for s in structured_candidates:
        if len(s) == 12 and sum(1 for c in s if c != 0) == 12 and all(-64 <= c <= 64 for c in s) and s[-1] != 0:
            try_poly(s)

    # Periodic patterns
    for period in [2, 3, 4, 6]:
        for base in range(-4, 5):
            for amplitude in range(1, 5):
                pattern = []
                for i in range(12):
                    val = base + amplitude * ((i % period) - period // 2)
                    if val == 0:
                        val = 1
                    pattern.append(val)
                if all(-64 <= c <= 64 for c in pattern) and pattern[-1] != 0:
                    try_poly(pattern)

    # Block-sign patterns: first k positive, next k negative, fill rest
    for k in range(2, 7):
        for sign in [1, -1]:
            poly = [0] * 12
            for i in range(min(k, 12)):
                poly[i] = sign
            for i in range(k, min(2 * k, 12)):
                poly[i] = -sign
            for i in range(2 * k, 12):
                if poly[i] == 0:
                    poly[i] = sign * (1 + (i % 3))
            if sum(1 for c in poly if c != 0) == 12 and poly[-1] != 0:
                try_poly(poly)

    # ------------------------------------------------------------------
    # REFINEMENT MACHINERY (from current program): targeted local search
    # with cancellation-aware candidate values, then exhaustive single
    # coefficient scan on the best.
    # ------------------------------------------------------------------
    random.seed(42)

    def local_search(start, max_iters=2000):
        curr = list(start)
        curr_score = evaluate(curr)

        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            improved = False

            # Move 1: change one coefficient to a value related to others
            pos = random.randint(0, 11)
            old_val = curr[pos]
            candidate_vals = set()
            for op in range(12):
                if op != pos:
                    candidate_vals.add(curr[op])
                    candidate_vals.add(-curr[op])
                    candidate_vals.add(2 * curr[op])
                    candidate_vals.add(-2 * curr[op])
            for _ in range(4):
                candidate_vals.add(random.randint(-16, 16))
            candidate_vals = [v for v in candidate_vals if -64 <= v <= 64 and v != 0 and v != old_val]
            if candidate_vals:
                new_val = random.choice(candidate_vals)
                curr[pos] = new_val
                new_score = evaluate(curr)
                if new_score < curr_score:
                    curr_score = new_score
                    improved = True
                    try_poly(curr)
                else:
                    curr[pos] = old_val

            # Move 2: swap two coefficients
            if random.random() < 0.2:
                i, j = random.sample(range(12), 2)
                a, b = curr[i], curr[j]
                curr[i], curr[j] = b, a
                new_score = evaluate(curr)
                if new_score < curr_score:
                    curr_score = new_score
                    improved = True
                    try_poly(curr)
                else:
                    curr[i], curr[j] = a, b

            # Move 3: negate a coefficient
            if random.random() < 0.1:
                pos = random.randint(0, 11)
                old_val = curr[pos]
                if old_val != 0:
                    curr[pos] = -old_val
                    new_score = evaluate(curr)
                    if new_score < curr_score:
                        curr_score = new_score
                        improved = True
                        try_poly(curr)
                    else:
                        curr[pos] = old_val

            if not improved and random.random() < 0.04:
                for _ in range(random.randint(1, 3)):
                    p = random.randint(0, 11)
                    curr[p] = random.randint(-16, 16)
                    if curr[p] == 0:
                        curr[p] = 1

        return curr, curr_score

    # Restart loop
    for restart in range(400):
        if time.time() > deadline - 12:
            break

        if restart < 8:
            start = list(best_poly)
        elif random.random() < 0.5:
            start = list(best_poly)
            for _ in range(random.randint(1, 3)):
                p = random.randint(0, 11)
                start[p] = max(-64, min(64, start[p] + random.randint(-8, 8)))
                if start[p] == 0:
                    start[p] = 1
        else:
            pt = random.randint(0, 3)
            if pt == 0:
                start = [random.choice([-1, 1]) * random.randint(1, 6) for _ in range(12)]
            elif pt == 1:
                base = random.randint(1, 4)
                start = [base * random.choice([-1, 1]) * (1 + (i % 3)) for i in range(12)]
            elif pt == 2:
                start = [random.randint(-4, 4) for _ in range(12)]
                for i in range(12):
                    if start[i] == 0:
                        start[i] = random.choice([-1, 1])
            else:
                start = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]

        start = [max(-64, min(64, c)) for c in start]
        for i in range(12):
            if start[i] == 0:
                start[i] = 1

        result, score = local_search(start, max_iters=1200)
        if score < best_score:
            best_score = score
            best_poly = result

    # Final exhaustive single-coefficient refinement on best
    if time.time() < deadline - 2:
        for pos in range(12):
            if time.time() > deadline - 1:
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

    # Ensure validity of final answer
    if sum(1 for c in best_poly if c != 0) != 12 or any(c < -64 or c > 64 for c in best_poly) or best_poly[-1] == 0:
        best_poly = [1] * 12

    return best_poly
