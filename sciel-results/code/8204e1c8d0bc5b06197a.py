"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _hamming_dist(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def _covered_by(word, codewords, radius):
    for cw in codewords:
        if _hamming_dist(word, cw) <= radius:
            return True
    return False


def _build_coverage(codewords, q, n, radius):
    """Return a set of all words covered by the given codewords."""
    covered = set()
    for cw in codewords:
        # Add the codeword itself
        covered.add(tuple(cw))
        if radius >= 1:
            for pos in range(n):
                orig = cw[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    lst = list(cw)
                    lst[pos] = new_val
                    covered.add(tuple(lst))
    return covered


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    total_words = q ** n

    # Precompute all words as indices
    def word_to_idx(word):
        idx = 0
        for v in word:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute covered indices for a codeword (as a numpy array for speed)
    def get_covered_indices(cw):
        """Return a numpy array of indices of words covered by cw."""
        indices = [word_to_idx(cw)]
        if radius >= 1:
            for pos in range(n):
                orig = cw[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    lst = list(cw)
                    lst[pos] = new_val
                    indices.append(word_to_idx(lst))
        return np.array(indices, dtype=np.int32)

    # Build coverage count using numpy for speed
    def build_coverage_count(cw_list):
        counts = np.zeros(total_words, dtype=np.int32)
        for cw in cw_list:
            covered_indices = get_covered_indices(cw)
            np.add.at(counts, covered_indices, 1)
        return counts

    # Check if removing a codeword keeps coverage valid
    def is_redundant(cw, counts):
        covered_indices = get_covered_indices(cw)
        # If any covered word has count <= 1, it's only covered by this codeword
        return np.all(counts[covered_indices] > 1)

    # Initial construction: direct sum of ternary tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n >= 4:
        initial_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    else:
        # Fallback: start with all words
        initial_codewords = [list(word) for word in itertools.product(range(q), repeat=n)]

    best_codewords = None
    best_len = len(initial_codewords)

    # Multiple restarts with different strategies
    for restart in range(20):
        if time.time() > deadline:
            break

        codewords = list(initial_codewords)

        # Phase 1: Greedy removal (multiple passes)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            counts = build_coverage_count(codewords)

            # Try removing in reverse order
            for i in range(len(codewords) - 1, -1, -1):
                if time.time() > deadline:
                    break
                if is_redundant(codewords[i], counts):
                    codewords.pop(i)
                    improved = True
                    counts = build_coverage_count(codewords)
                    break

            if not improved and time.time() < deadline:
                # Try forward order
                counts = build_coverage_count(codewords)
                for i in range(len(codewords)):
                    if time.time() > deadline:
                        break
                    if is_redundant(codewords[i], counts):
                        codewords.pop(i)
                        improved = True
                        counts = build_coverage_count(codewords)
                        break

            if not improved and time.time() < deadline:
                # Try random order
                counts = build_coverage_count(codewords)
                indices = list(range(len(codewords)))
                random.shuffle(indices)
                for i in indices:
                    if time.time() > deadline:
                        break
                    if is_redundant(codewords[i], counts):
                        codewords.pop(i)
                        improved = True
                        counts = build_coverage_count(codewords)
                        break

        # Phase 2: Swap-based local search (add-then-remove)
        if time.time() < deadline:
            counts = build_coverage_count(codewords)
            swap_improved = True
            max_swap_iters = 50
            for _ in range(max_swap_iters):
                if time.time() > deadline:
                    break
                if not swap_improved:
                    break
                swap_improved = False

                # Try to find a swap: remove one, add another, then try to remove more
                counts = build_coverage_count(codewords)
                # Find codewords that are "least useful" (cover fewest uniquely-covered words)
                unique_counts = []
                for i, cw in enumerate(codewords):
                    covered = get_covered_indices(cw)
                    unique = np.sum(counts[covered] == 1)
                    unique_counts.append((unique, i))

                # Sort by unique coverage (least first)
                unique_counts.sort()

                # Try swapping the least useful codeword with a random new one
                for unique, i in unique_counts[:5]:  # Try top 5 least useful
                    if time.time() > deadline:
                        break
                    if unique > 0:
                        continue  # Can't remove this one directly

                    # Try adding a random codeword and see if we can then remove more
                    new_cw = [random.randint(0, q - 1) for _ in range(n)]
                    if new_cw in codewords:
                        continue

                    # Temporarily add and remove
                    test_cw = codewords.copy()
                    test_cw.pop(i)
                    test_cw.append(new_cw)

                    # Check if this is still valid and allows further removals
                    test_counts = build_coverage_count(test_cw)
                    if np.any(test_counts == 0):
                        continue  # Not valid

                    # Try to remove more from the test set
                    test_improved = True
                    while test_improved and time.time() < deadline:
                        test_improved = False
                        test_counts = build_coverage_count(test_cw)
                        for j in range(len(test_cw) - 1, -1, -1):
                            if time.time() > deadline:
                                break
                            if is_redundant(test_cw[j], test_counts):
                                test_cw.pop(j)
                                test_improved = True
                                test_counts = build_coverage_count(test_cw)
                                break

                    if len(test_cw) < len(codewords):
                        codewords = test_cw
                        swap_improved = True
                        break

        # Update best
        if len(codewords) < best_len:
            best_len = len(codewords)
            best_codewords = codewords.copy()

        # If we're already at a good size, try more aggressive search
        if best_len <= 75 and time.time() < deadline:
            # Try a few more restarts with perturbation
            for pert in range(5):
                if time.time() > deadline:
                    break
                perturbed = best_codewords.copy()
                # Remove a random codeword and try to re-add a better one
                if len(perturbed) > 10:
                    remove_idx = random.randint(0, len(perturbed) - 1)
                    perturbed.pop(remove_idx)

                    # Try to fix coverage by adding random codewords
                    for _ in range(20):
                        if time.time() > deadline:
                            break
                        counts = build_coverage_count(perturbed)
                        if np.all(counts > 0):
                            break  # Still valid
                        # Find uncovered words
                        uncovered_indices = np.where(counts == 0)[0]
                        if len(uncovered_indices) == 0:
                            break
                        # Add a codeword near an uncovered word
                        unc_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
                        unc_word = idx_to_word(unc_idx)
                        new_cw = [random.randint(0, q - 1) for _ in range(n)]
                        # Bias toward covering the uncovered word
                        pos = random.randint(0, n - 1)
                        new_cw[pos] = unc_word[pos]
                        if new_cw not in perturbed:
                            perturbed.append(new_cw)

                    # Try to remove redundant codewords
                    improved = True
                    while improved and time.time() < deadline:
                        improved = False
                        counts = build_coverage_count(perturbed)
                        for i in range(len(perturbed) - 1, -1, -1):
                            if time.time() > deadline:
                                break
                            if is_redundant(perturbed[i], counts):
                                perturbed.pop(i)
                                improved = True
                                counts = build_coverage_count(perturbed)
                                break

                    # Check validity
                    final_counts = build_coverage_count(perturbed)
                    if np.all(final_counts > 0) and len(perturbed) < best_len:
                        best_len = len(perturbed)
                        best_codewords = perturbed.copy()

    if best_codewords is None:
        best_codewords = initial_codewords

    return best_codewords
# EVOLVE-BLOCK-END
