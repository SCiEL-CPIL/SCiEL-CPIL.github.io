import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # Fallback for other parameters
    import itertools
    total = q ** n
    if total <= 10000:
        all_words = [list(c) for c in itertools.product(range(q), repeat=n)]
        uncovered = set(range(len(all_words)))
        codewords = []

        def hamming_dist(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)

        while uncovered:
            if time.time() > deadline - 5.0:
                break
            best_idx = -1
            best_count = -1
            for i in range(len(all_words)):
                count = 0
                for j in uncovered:
                    if hamming_dist(all_words[i], all_words[j]) <= radius:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx == -1:
                break
            codewords.append(all_words[best_idx])
            new_uncovered = set()
            for j in uncovered:
                if hamming_dist(all_words[best_idx], all_words[j]) > radius:
                    new_uncovered.add(j)
            uncovered = new_uncovered

        return codewords
    else:
        return [list(c) for c in itertools.product(range(q), repeat=n)]


def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17

    # Precompute base-3 weights
    weights = [3 ** (n - 1 - i) for i in range(n)]

    # Precompute balls: balls[w, 0..16] are the 17 words within Hamming distance 1 of w
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1

    # Precompute ball_membership: ball_membership[w, v] = True if v in balls[w]
    ball_membership = np.zeros((total, total), dtype=np.int8)
    for w in range(total):
        ball_membership[w, balls[w]] = 1

    best_size = total
    best_set = None

    # Helper: convert index to digits
    def idx_to_digits(idx):
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = idx % q
            idx //= q
        return digits

    # Phase 1: Iterated Local Search (ILS) with perturbation and repair
    # Start with a good initial solution, then iterate:
    #   - Perturb the current solution (random swaps/removals)
    #   - Repair to validity (greedy add)
    #   - Local search (removal, swap, 2-exchange)
    #   - Accept if improved

    def local_search(selected, cover_count, deadline):
        """Run local search until no improvement. Returns (selected, cover_count, improved)."""
        improved_total = True
        while improved_total and time.time() < deadline - 5.0:
            improved_total = False

            # 3a: Remove redundant codewords
            changed = True
            while changed and time.time() < deadline - 5.0:
                changed = False
                new_selected = []
                for c in selected:
                    if np.min(cover_count[balls[c]]) > 1:
                        cover_count[balls[c]] -= 1
                        changed = True
                    else:
                        new_selected.append(c)
                if changed:
                    improved_total = True
                selected = new_selected

            # 3b: Swap phase
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                new_selected = []
                for c in selected:
                    # Try to remove
                    if np.min(cover_count[balls[c]]) > 1:
                        cover_count[balls[c]] -= 1
                        improved = True
                        continue

                    # Critical words: covered only by c
                    critical_mask = (cover_count[balls[c]] == 1)
                    critical_words = balls[c][critical_mask]

                    if len(critical_words) == 0:
                        cover_count[balls[c]] -= 1
                        improved = True
                        continue

                    # Find candidates c' that cover all critical words
                    candidate_mask = np.ones(total, dtype=bool)
                    for cw in critical_words:
                        candidate_mask &= (ball_membership[:, cw] == 1)

                    valid_candidates = np.where(candidate_mask)[0]
                    valid_candidates = valid_candidates[valid_candidates != c]

                    if len(valid_candidates) > 0:
                        c_prime = valid_candidates[np.random.randint(len(valid_candidates))]
                        cover_count[balls[c]] -= 1
                        cover_count[balls[c_prime]] += 1
                        new_selected.append(c_prime)
                        improved = True
                    else:
                        new_selected.append(c)

                if improved:
                    improved_total = True
                selected = new_selected

            # 3c: 2-Exchange (Remove 2, Add 1)
            if len(selected) > 450 and time.time() < deadline - 8.0:
                improved_2ex = True
                while improved_2ex and time.time() < deadline - 5.0:
                    improved_2ex = False
                    new_selected = list(selected)

                    indices = list(range(len(new_selected)))
                    np.random.shuffle(indices)

                    for i in range(len(indices)):
                        for j in range(i + 1, len(indices)):
                            if time.time() > deadline - 5.0:
                                break

                            c1 = new_selected[indices[i]]
                            c2 = new_selected[indices[j]]

                            words_c1 = balls[c1]
                            words_c2 = balls[c2]

                            # Find words that become uncovered if we remove c1 and c2
                            to_cover = set()
                            for w in words_c1:
                                if cover_count[w] == 1:
                                    to_cover.add(w)
                            for w in words_c2:
                                if cover_count[w] == 1:
                                    to_cover.add(w)
                            set_c1 = set(words_c1.tolist())
                            set_c2 = set(words_c2.tolist())
                            inter = set_c1 & set_c2
                            for w in inter:
                                if cover_count[w] == 2:
                                    to_cover.add(w)

                            if not to_cover:
                                # Both are redundant
                                cover_count[balls[c1]] -= 1
                                cover_count[balls[c2]] -= 1
                                new_selected = [c for c in new_selected if c != c1 and c != c2]
                                improved_2ex = True
                                break

                            # Find c3 that covers all words in to_cover
                            candidate_mask = np.ones(total, dtype=bool)
                            for w in to_cover:
                                candidate_mask &= (ball_membership[:, w] == 1)

                            valid_candidates = np.where(candidate_mask)[0]
                            valid_candidates = valid_candidates[(valid_candidates != c1) & (valid_candidates != c2)]

                            if len(valid_candidates) > 0:
                                c3 = valid_candidates[np.random.randint(len(valid_candidates))]
                                cover_count[balls[c1]] -= 1
                                cover_count[balls[c2]] -= 1
                                cover_count[balls[c3]] += 1
                                new_selected = [c for c in new_selected if c != c1 and c != c2]
                                new_selected.append(c3)
                                improved_2ex = True
                                break

                        if improved_2ex:
                            break
                    if improved_2ex:
                        improved_total = True
                        selected = new_selected

        return selected, cover_count

    def repair(selected, cover_count, deadline):
        """Add codewords greedily until all words are covered."""
        uncovered_mask = (cover_count == 0)
        while np.any(uncovered_mask) and time.time() < deadline - 5.0:
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
        return selected, cover_count

    # Initial solution: greedy set cover
    cover_count = np.zeros(total, dtype=np.int32)
    selected = []
    uncovered_mask = np.ones(total, dtype=bool)

    while np.any(uncovered_mask) and time.time() < deadline - 10.0:
        counts = uncovered_mask[balls].sum(axis=1)
        max_count = np.max(counts)
        if max_count == 0:
            break
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[np.random.randint(len(candidates))]
        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        uncovered_mask = (cover_count == 0)

    # Local search on initial solution
    if not np.any(uncovered_mask):
        selected, cover_count = local_search(selected, cover_count, deadline)
        best_size = len(selected)
        best_set = selected[:]

    # Iterated Local Search
    ilswindow = 0
    for ilswindow in range(100):
        if time.time() > deadline - 15.0:
            break

        # Perturbation: random swaps and removals
        perturbed = list(selected)
        perturbed_count = cover_count.copy()

        # Perform k random perturbations
        k_perturb = min(5, len(perturbed) // 10)
        for _ in range(k_perturb):
            if time.time() > deadline - 10.0:
                break
            if len(perturbed) < 2:
                break
            # Random swap: remove one, add another
            idx_to_remove = np.random.randint(len(perturbed))
            c_remove = perturbed[idx_to_remove]
            perturbed_count[balls[c_remove]] -= 1
            perturbed.pop(idx_to_remove)

            # Add a random codeword
            c_add = np.random.randint(total)
            if c_add not in perturbed:
                perturbed.append(c_add)
                perturbed_count[balls[c_add]] += 1

        # Repair to validity
        perturbed, perturbed_count = repair(perturbed, perturbed_count, deadline)

        # Local search
        perturbed, perturbed_count = local_search(perturbed, perturbed_count, deadline)

        # Accept if improved
        if len(perturbed) < best_size:
            best_size = len(perturbed)
            best_set = perturbed[:]
            cover_count = perturbed_count
            selected = perturbed[:]

            if best_size <= 486:
                break

    if best_set is None:
        # Fallback: return a valid (large) code
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        uncovered_mask = np.ones(total, dtype=bool)

        while np.any(uncovered_mask) and time.time() < deadline - 5.0:
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)

        best_set = selected

    # Convert to digit lists
    result = [idx_to_digits(cw) for cw in best_set]

    # Ensure distinctness
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)

    return unique_result
