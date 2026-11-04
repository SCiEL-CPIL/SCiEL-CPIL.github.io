"""Covering code K_3(7,1) via Lagrangian-guided greedy + numpy local refinement."""
import time
import random
import itertools
import numpy as np

_DEADLINE = 140.0  # safety margin before 150s kill


def construct(q, n, radius):
    t0 = time.time()

    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters
        if n <= 4 and q == 3 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbors for each word (ball of radius 1)
    neighbors_list = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))

    # Also keep as Python sets for fast membership
    neighbor_sets = [set(nb.tolist()) for nb in neighbors_list]

    # ---------- Lagrangian-guided Greedy Construction ----------
    # We model this as minimum weight dominating set.
    # Lagrangian relaxation: dual variables lambda_v >= 0 for each vertex v.
    # Reduced cost for node u: r_u = 1 - sum_{v in N(u)} lambda_v
    # We want to pick nodes with most negative reduced cost (i.e., highest sum of lambda_v in their neighborhood).
    # Subgradient update: lambda_v += step * (1 - sum_{u in S, v in N(u)} 1)
    #   where S is current solution. If v is uncovered, gradient is +1 (increase lambda_v).
    #   If v is covered, gradient is -1 (decrease lambda_v).

    def lagrangian_greedy(seed):
        rng = random.Random(seed)
        lambda_vec = np.zeros(N, dtype=np.float64)
        coverage = np.zeros(N, dtype=np.int32)
        uncovered_mask = np.ones(N, dtype=bool)
        code = []

        # Initial lambda: random small values to break symmetry
        lambda_vec += rng.uniform(0.01, 0.1, size=N)

        max_iter = 300  # max codewords to add
        for _ in range(max_iter):
            if time.time() - t0 > _DEADLINE - 20:
                break
            if not uncovered_mask.any():
                break

            # Compute reduced costs: for each candidate u, score = sum of lambda_v for v in N(u)
            # Only consider candidates that cover at least one uncovered word
            best_c = -1
            best_score = -np.inf
            best_uncov = 0

            # For efficiency, compute score vectorized
            # score[u] = lambda_vec[neighbors_list[u]].sum()
            # But we only care about candidates covering uncovered words
            # Let's compute for all, it's fast enough for N=2187
            scores = np.array([lambda_vec[nb].sum() for nb in neighbors_list])

            # Count uncovered covered by each candidate
            uncov_counts = np.array([np.sum(uncovered_mask[nb]) for nb in neighbors_list])

            # Filter: must cover at least 1 uncovered
            valid = uncov_counts > 0
            if not valid.any():
                break

            # Among valid, pick highest score. Tie-break by uncov_count.
            # Combine into a single metric: score + small bonus for coverage
            combined = scores + 0.001 * uncov_counts
            combined[~valid] = -np.inf

            best_c = int(np.argmax(combined))
            best_score = combined[best_c]
            best_uncov = int(uncov_counts[best_c])

            if best_uncov == 0:
                break

            # Add to code
            code.append(best_c)
            coverage[neighbors_list[best_c]] += 1
            uncovered_mask[neighbors_list[best_c]] = False

            # Subgradient update for lambda
            # gradient_v = 1 if v uncovered, -1 if covered
            # step size: use Polyak-like or simple decreasing
            step = 0.05 / (1 + len(code) * 0.1)
            grad = np.where(uncovered_mask, 1.0, -1.0)
            lambda_vec += step * grad
            lambda_vec = np.maximum(lambda_vec, 0.0)  # keep non-negative

        # If not fully covered, do greedy repair
        if uncovered_mask.any():
            # Simple greedy repair
            while uncovered_mask.any() and time.time() - t0 < _DEADLINE - 15:
                uncov_counts = np.array([np.sum(uncovered_mask[nb]) for nb in neighbors_list])
                # Prefer candidates not already in code
                in_code = np.zeros(N, dtype=bool)
                for c in code:
                    in_code[c] = True
                uncov_counts[in_code] = 0
                if uncov_counts.max() == 0:
                    # Allow re-adding? No, just pick best overall
                    uncov_counts = np.array([np.sum(uncovered_mask[nb]) for nb in neighbors_list])
                best_c = int(np.argmax(uncov_counts))
                if uncov_counts[best_c] == 0:
                    break
                code.append(best_c)
                coverage[neighbors_list[best_c]] += 1
                uncovered_mask[neighbors_list[best_c]] = False

        return code, coverage

    # ---------- Local Refinement ----------
    def try_remove(code, coverage, neighbors_list, t0):
        """Try to remove redundant codewords. Returns (new_code, new_coverage, removed_count)."""
        removed = 0
        for c in list(code):
            if time.time() - t0 > _DEADLINE - 10:
                break
            # Check if all neighbors of c are covered by other codewords
            if np.min(coverage[neighbors_list[c]]) > 1:
                code.remove(c)
                coverage[neighbors_list[c]] -= 1
                removed += 1
        return code, coverage, removed

    def kick(code, coverage, neighbors_list, neighbor_sets, t0, rng):
        """Remove one random codeword and repair. Accept if size decreases."""
        if len(code) < 2:
            return code, coverage, False

        old_size = len(code)
        idx_remove = rng.randint(0, len(code) - 1)
        c_remove = code[idx_remove]

        temp_coverage = coverage.copy()
        temp_coverage[neighbors_list[c_remove]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]

        if len(uncovered) == 0:
            # Can remove safely
            code.remove(c_remove)
            coverage[neighbors_list[c_remove]] -= 1
            return code, coverage, True

        uncovered_set = set(uncovered.tolist())
        temp_code = [c for c in code if c != c_remove]

        # Greedy repair
        added = []
        temp_uncov = set(uncovered_set)
        while temp_uncov and time.time() - t0 < _DEADLINE - 10:
            # Find best candidate to add
            best_c = -1
            best_cnt = 0
            # Candidate pool: neighbors of uncovered words
            candidate_pool = set()
            for w in temp_uncov:
                candidate_pool.update(neighbor_sets[w])
            # Remove already in temp_code
            temp_code_set = set(temp_code)
            candidate_pool -= temp_code_set

            for c in candidate_pool:
                cnt = 0
                for x in neighbors_list[c]:
                    if x in temp_uncov:
                        cnt += 1
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
                if best_cnt == len(temp_uncov):
                    break

            if best_c == -1 or best_cnt == 0:
                break

            added.append(best_c)
            for w in neighbors_list[best_c]:
                temp_uncov.discard(w)

        new_size = old_size - 1 + len(added)
        if new_size < old_size and not temp_uncov:
            # Accept
            code.remove(c_remove)
            coverage[neighbors_list[c_remove]] -= 1
            for c in added:
                code.append(c)
                coverage[neighbors_list[c]] += 1
            return code, coverage, True
        else:
            return code, coverage, False

    def kick2(code, coverage, neighbors_list, neighbor_sets, t0, rng):
        """Remove two random codewords and repair. Accept if size decreases."""
        if len(code) < 4:
            return code, coverage, False

        old_size = len(code)
        i1 = rng.randint(0, len(code) - 1)
        i2 = rng.randint(0, len(code) - 1)
        if i1 == i2:
            i2 = (i2 + 1) % len(code)
        c1 = code[i1]
        c2 = code[i2]

        temp_coverage = coverage.copy()
        temp_coverage[neighbors_list[c1]] -= 1
        temp_coverage[neighbors_list[c2]] -= 1

        uncovered = np.where(temp_coverage == 0)[0]
        if len(uncovered) == 0:
            code.remove(c1)
            code.remove(c2)
            coverage[neighbors_list[c1]] -= 1
            coverage[neighbors_list[c2]] -= 1
            return code, coverage, True

        uncovered_set = set(uncovered.tolist())
        temp_code = [c for c in code if c != c1 and c != c2]

        added = []
        temp_uncov = set(uncovered_set)
        while temp_uncov and time.time() - t0 < _DEADLINE - 10:
            best_c = -1
            best_cnt = 0
            candidate_pool = set()
            for w in temp_uncov:
                candidate_pool.update(neighbor_sets[w])
            temp_code_set = set(temp_code)
            candidate_pool -= temp_code_set

            for c in candidate_pool:
                cnt = 0
                for x in neighbors_list[c]:
                    if x in temp_uncov:
                        cnt += 1
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
                if best_cnt == len(temp_uncov):
                    break

            if best_c == -1 or best_cnt == 0:
                break

            added.append(best_c)
            for w in neighbors_list[best_c]:
                temp_uncov.discard(w)

        new_size = old_size - 2 + len(added)
        if new_size < old_size and not temp_uncov:
            code.remove(c1)
            code.remove(c2)
            coverage[neighbors_list[c1]] -= 1
            coverage[neighbors_list[c2]] -= 1
            for c in added:
                code.append(c)
                coverage[neighbors_list[c]] += 1
            return code, coverage, True
        else:
            return code, coverage, False

    # ---------- Main Search Loop ----------
    best_code = None
    best_size = N + 1

    rng_main = random.Random(12345)

    # Try multiple Lagrangian greedy constructions with different seeds
    for seed in range(50):
        if time.time() - t0 > _DEADLINE - 30:
            break

        code, coverage = lagrangian_greedy(seed)

        if not code:
            continue

        # Verify full coverage
        cov_check = np.zeros(N, dtype=np.int32)
        for c in code:
            cov_check[neighbors_list[c]] += 1
        if np.min(cov_check) == 0:
            # Not fully covered, skip or repair
            # Try quick repair
            uncovered = np.where(cov_check == 0)[0]
            for w in uncovered:
                # Add a neighbor of w
                best_c = -1
                best_cnt = 0
                for c in neighbor_sets[w]:
                    cnt = np.sum(cov_check[neighbors_list[c]] == 0)
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_c = c
                if best_c != -1:
                    code.append(best_c)
                    cov_check[neighbors_list[best_c]] += 1
            # Re-verify
            cov_check = np.zeros(N, dtype=np.int32)
            for c in code:
                cov_check[neighbors_list[c]] += 1
            if np.min(cov_check) == 0:
                continue

        coverage = cov_check
        sz = len(code)
        if sz < best_size:
            best_size = sz
            best_code = list(code)

        # Local refinement
        if time.time() - t0 > _DEADLINE - 15:
            break

        # Exhaustive removal
        improved = True
        while improved and time.time() - t0 < _DEADLINE - 10:
            code, coverage, removed = try_remove(code, coverage, neighbors_list, t0)
            improved = removed > 0
            if removed > 0 and len(code) < best_size:
                best_size = len(code)
                best_code = list(code)

        # Kicks
        kicks_tried = 0
        while kicks_tried < 30 and time.time() - t0 < _DEADLINE - 5:
            success = False
            if rng_main.random() < 0.5:
                code, coverage, success = kick(code, coverage, neighbors_list, neighbor_sets, t0, rng_main)
            else:
                code, coverage, success = kick2(code, coverage, neighbors_list, neighbor_sets, t0, rng_main)

            if success:
                kicks_tried = 0
                # After a successful kick, try removal again
                improved = True
                while improved and time.time() - t0 < _DEADLINE - 5:
                    code, coverage, removed = try_remove(code, coverage, neighbors_list, t0)
                    improved = removed > 0
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
            else:
                kicks_tried += 1

    if best_code is None:
        # Fallback: direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Convert indices to word lists
    result = []
    for idx in best_code:
        result.append(idx_to_tuple(idx))
    return result
