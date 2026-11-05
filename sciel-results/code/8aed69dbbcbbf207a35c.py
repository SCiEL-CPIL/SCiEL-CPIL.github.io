import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0

    def get_direct_sum():
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [[0] * n]

    if q == 3 and n == 6 and radius == 1:
        total = 3 ** n
        powers = [3 ** p for p in range(n)]

        # Precompute neighbors: each word covers itself + 2*6 = 13 neighbors
        neighbors = np.empty((total, 13), dtype=np.int32)
        for i in range(total):
            digits = [(i // powers[p]) % 3 for p in range(n)]
            neighbors[i, 0] = i
            k = 1
            for p in range(n):
                for alt in range(3):
                    if alt != digits[p]:
                        neighbors[i, k] = i + (alt - digits[p]) * powers[p]
                        k += 1

        neighbor_sets = [set(neighbors[i].tolist()) for i in range(total)]

        def idx_to_list(idx):
            return [(idx // powers[p]) % 3 for p in range(n)]

        # Phase 1: Fast greedy set cover
        uncovered = np.ones(total, dtype=np.int8)
        codewords = []

        while uncovered.sum() > 0:
            if time.time() > deadline - 60:
                break
            counts = np.sum(uncovered[neighbors], axis=1)
            best_c = int(np.argmax(counts))
            if counts[best_c] == 0:
                break
            codewords.append(best_c)
            uncovered[neighbors[best_c]] = 0

        # Ensure full coverage with random fallback
        if uncovered.sum() != 0:
            random.seed(42)
            np.random.seed(42)
            while uncovered.sum() > 0:
                c = int(np.random.randint(total))
                codewords.append(c)
                uncovered[neighbors[c]] = 0

        # Build coverage counts
        coverage_counts = np.zeros(total, dtype=np.int32)
        for c in codewords:
            coverage_counts[neighbors[c]] += 1

        in_set = set(codewords)
        best_codewords = list(codewords)
        best_size = len(codewords)

        random.seed(42)
        np.random.seed(42)

        # Phase 2: Local search with simulated annealing
        T = 2.0
        T_min = 0.01
        cooling = 0.9998
        no_improve = 0
        max_no_improve = 12000

        while time.time() < deadline - 15.0 and no_improve < max_no_improve:
            # Step 1: Try to remove redundant codewords
            redundant_indices = []
            for idx, c in enumerate(codewords):
                if coverage_counts[neighbors[c]].min() >= 2:
                    redundant_indices.append(idx)

            if redundant_indices:
                idx_to_remove = random.choice(redundant_indices)
                c_to_remove = codewords[idx_to_remove]
                codewords.pop(idx_to_remove)
                coverage_counts[neighbors[c_to_remove]] -= 1
                in_set.discard(c_to_remove)

                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                    no_improve = 0
                else:
                    no_improve += 1
                T = max(T * cooling, T_min)
                continue

            # Step 2: No redundant words. Try swaps.
            found_swap = False
            num_tries = 8
            for _ in range(num_tries):
                idx_to_remove = random.randint(0, len(codewords) - 1)
                c_to_remove = codewords[idx_to_remove]

                # Critical words: neighbors of c_to_remove with coverage exactly 1
                crit_mask = (coverage_counts[neighbors[c_to_remove]] == 1)
                critical_words = neighbors[c_to_remove][crit_mask]

                if len(critical_words) == 0:
                    continue

                # Candidates must cover ALL critical words
                w0 = int(critical_words[0])
                candidates = neighbors[w0]

                valid_candidates = []
                for cand in candidates:
                    if cand == c_to_remove:
                        continue
                    if all(w in neighbor_sets[int(cand)] for w in critical_words):
                        valid_candidates.append(int(cand))

                if not valid_candidates:
                    continue

                # Pick best candidate: maximize minimum coverage among its neighbors
                best_cand = valid_candidates[0]
                best_min_cov = -1
                for cand in valid_candidates:
                    min_cov = int(coverage_counts[neighbors[cand]].min())
                    if min_cov > best_min_cov:
                        best_min_cov = min_cov
                        best_cand = cand

                # Perform swap
                codewords[idx_to_remove] = best_cand
                coverage_counts[neighbors[c_to_remove]] -= 1
                coverage_counts[neighbors[best_cand]] += 1
                in_set.discard(c_to_remove)
                in_set.add(best_cand)

                found_swap = True
                break

            if not found_swap:
                no_improve += 1
            else:
                no_improve = 0

            T = max(T * cooling, T_min)

        # Final cleanup: iteratively remove redundant words
        improved = True
        while improved:
            improved = False
            for idx in range(len(codewords)):
                c = codewords[idx]
                if coverage_counts[neighbors[c]].min() >= 2:
                    codewords.pop(idx)
                    coverage_counts[neighbors[c]] -= 1
                    improved = True
                    if len(codewords) < best_size:
                        best_size = len(codewords)
                        best_codewords = list(codewords)
                    break

        # Use the best found
        if len(best_codewords) < len(codewords):
            codewords = best_codewords
        elif len(codewords) < len(best_codewords):
            best_codewords = list(codewords)

        # Validate
        covered = np.zeros(total, dtype=np.int8)
        for c in codewords:
            covered[neighbors[c]] = 1
        if covered.all():
            final_codewords = [idx_to_list(c) for c in codewords]
            ds = get_direct_sum()
            if len(final_codewords) <= len(ds):
                return final_codewords
            else:
                return ds

        ds = get_direct_sum()
        return ds

    return get_direct_sum()
