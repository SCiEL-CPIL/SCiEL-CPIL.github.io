import itertools
import time
import random
import math
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s hard kill

    if q == 3 and n == 7 and radius == 1:
        # ---- Precompute all words, index map, and balls (each ball = 15 words) ----
        all_words = list(itertools.product(range(3), repeat=7))
        N = len(all_words)  # 2187
        word_to_idx = {w: i for i, w in enumerate(all_words)}

        balls = [None] * N
        for w in all_words:
            covered_indices = [word_to_idx[w]]
            wl = list(w)
            for i in range(7):
                orig = wl[i]
                for val in range(3):
                    if val != orig:
                        wl[i] = val
                        covered_indices.append(word_to_idx[tuple(wl)])
                wl[i] = orig
            balls[i] = covered_indices

        random.seed(42)

        # ---- 1. Greedy set cover ----
        covered = np.zeros(N, dtype=bool)
        codewords = []
        while not covered.all() and time.time() < deadline - 60:
            best_cw = -1
            best_count = -1
            for i in range(N):
                b = balls[i]
                new_cov = np.sum(~covered[b])
                if new_cov > best_count:
                    best_count = new_cov
                    best_cw = i
            if best_count == 0:
                break
            codewords.append(best_cw)
            covered[balls[best_cw]] = True

        # ---- 2. Initial coverage counts ----
        coverage_count = np.zeros(N, dtype=np.int32)
        for cw in codewords:
            coverage_count[balls[cw]] += 1

        # ---- 3. Redundancy removal pass ----
        def do_redundancy_removal(cw_list, cc):
            i = 0
            while i < len(cw_list):
                b = balls[cw_list[i]]
                if np.min(cc[b]) >= 2:
                    cc[b] -= 1
                    cw_list.pop(i)
                else:
                    i += 1
            return cw_list, cc

        codewords, coverage_count = do_redundancy_removal(codewords, coverage_count)

        # ---- 4. Simulated Annealing with FAST O(affected) swap evaluation ----
        # cost = size + P * uncovered ; P large enough to strongly prefer covering
        P = 100.0
        T_init = 3.0
        T_min = 0.01
        cooling = 0.9997
        restart_threshold = 4000

        best_size = len(codewords)
        best_codewords = list(codewords)
        best_coverage = coverage_count.copy()
        best_uncovered = int(np.sum(coverage_count == 0))
        best_cost = best_size + P * best_uncovered

        current_size = len(codewords)
        uncovered = int(np.sum(coverage_count == 0))
        steps_since_improvement = 0
        T = T_init

        in_code = set(codewords)

        while time.time() < deadline - 6:
            r = random.random()

            if r < 0.45:
                # ---- Remove (size -1) ----
                if not codewords:
                    T = max(T * cooling, T_min)
                    continue
                idx = random.randint(0, len(codewords) - 1)
                cw = codewords[idx]
                b = balls[cw]
                if np.min(coverage_count[b]) >= 2:
                    coverage_count[b] -= 1
                    codewords.pop(idx)
                    in_code.discard(cw)
                    current_size -= 1
                    uncovered = int(np.sum(coverage_count == 0))
                    steps_since_improvement = 0
                    cost = current_size + P * uncovered
                    if cost < best_cost:
                        best_cost = cost
                        best_size = current_size
                        best_uncovered = uncovered
                        best_codewords = list(codewords)
                        best_coverage = coverage_count.copy()
                # else: invalid removal, skip (no state change)

            elif r < 0.95:
                # ---- Swap (size unchanged) : FAST delta evaluation ----
                if not codewords:
                    T = max(T * cooling, T_min)
                    continue
                idx_old = random.randint(0, len(codewords) - 1)
                cw_old = codewords[idx_old]
                cw_new = random.randint(0, N - 1)
                if cw_new == cw_old or cw_new in in_code:
                    T = max(T * cooling, T_min)
                    continue

                b_old = balls[cw_old]
                b_new = balls[cw_new]

                # affected words = union of the two balls (<= 30 words)
                affected = set(b_old)
                affected.update(b_new)

                # compute delta in covered count (words going 0->1 or 1->0)
                delta_covered = 0
                for nb in affected:
                    old_c = coverage_count[nb]
                    new_c = old_c
                    if nb in b_old:
                        new_c -= 1
                    if nb in b_new:
                        new_c += 1
                    if old_c == 0 and new_c > 0:
                        delta_covered += 1
                    elif old_c > 0 and new_c == 0:
                        delta_covered -= 1

                new_uncovered = uncovered - delta_covered
                delta_cost = P * (-delta_covered)  # size unchanged

                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                    # accept: commit the change
                    coverage_count[b_old] -= 1
                    coverage_count[b_new] += 1
                    codewords[idx_old] = cw_new
                    in_code.discard(cw_old)
                    in_code.add(cw_new)
                    uncovered = new_uncovered
                    if delta_covered > 0:
                        steps_since_improvement = 0
                    elif delta_covered == 0:
                        steps_since_improvement = min(steps_since_improvement, restart_threshold // 2)
                # else: reject, no state change

            else:
                # ---- Add (size +1) : rarely, to escape deep local minima ----
                if codewords:
                    cw_new = random.randint(0, N - 1)
                    if cw_new not in in_code:
                        b_new = balls[cw_new]
                        # delta covered from adding
                        delta_covered = 0
                        for nb in b_new:
                            if coverage_count[nb] == 0:
                                delta_covered += 1
                        new_uncovered = uncovered - delta_covered
                        delta_cost = 1.0 + P * (-delta_covered)
                        if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                            coverage_count[b_new] += 1
                            codewords.append(cw_new)
                            in_code.add(cw_new)
                            current_size += 1
                            uncovered = new_uncovered

            T = max(T * cooling, T_min)
            steps_since_improvement += 1

            # ---- Restart if stuck ----
            if steps_since_improvement >= restart_threshold:
                codewords = list(best_codewords)
                coverage_count = best_coverage.copy()
                in_code = set(codewords)
                current_size = best_size
                uncovered = best_uncovered
                T = T_init * 0.6
                steps_since_improvement = 0

        # ---- 5. Final polish: exhaustive redundancy removal on the best solution ----
        final_code = list(best_codewords)
        final_cc = best_coverage.copy()
        do_redundancy_removal(final_code, final_cc)
        # a second pass can catch newly-exposed redundancies
        do_redundancy_removal(final_code, final_cc)

        if len(final_code) <= best_size:
            return [list(all_words[cw]) for cw in final_code]
        return [list(all_words[cw]) for cw in best_codewords]

    # ---- General fallback (not optimized for this specific case) ----
    all_words = list(itertools.product(range(q), repeat=n))

    def get_coverage_general(codeword):
        covered = set()
        covered.add(codeword)
        for i in range(n):
            for val in range(q):
                if val != codeword[i]:
                    new_word = list(codeword)
                    new_word[i] = val
                    covered.add(tuple(new_word))
        return covered

    codewords = []
    covered = set()
    while time.time() < deadline - 10:
        best_cw = None
        best_count = -1
        best_new_covered = set()
        for candidate in all_words:
            new_covered = get_coverage_general(candidate) - covered
            count = len(new_covered)
            if count > best_count:
                best_count = count
                best_cw = candidate
                best_new_covered = new_covered
        if best_count <= 0:
            break
        codewords.append(best_cw)
        covered.update(best_new_covered)
        if len(covered) == len(all_words):
            break

    i = 0
    while i < len(codewords) and time.time() < deadline - 5:
        cw_to_remove = codewords[i]
        coverage_of_cw = get_coverage_general(cw_to_remove)
        can_remove = True
        for w in coverage_of_cw:
            is_covered_by_other = False
            for j, other_cw in enumerate(codewords):
                if i == j:
                    continue
                if w in get_coverage_general(other_cw):
                    is_covered_by_other = True
                    break
            if not is_covered_by_other:
                can_remove = False
                break
        if can_remove:
            codewords.pop(i)
        else:
            i += 1

    return [list(cw) for cw in codewords]
