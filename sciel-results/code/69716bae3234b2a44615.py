"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


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

    # Precompute covered indices for each possible codeword index
    # Use a 2D array: covered_indices_map[cw_idx] = array of covered word indices
    all_cw_indices = list(range(total_words))
    
    # For each codeword, precompute which words it covers
    # This is expensive for large spaces, but for q=3, n=6, total=729, it's manageable
    covered_map = {}
    for cw_idx in range(total_words):
        cw = idx_to_word(cw_idx)
        indices = [cw_idx]
        if radius >= 1:
            for pos in range(n):
                orig = cw[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    lst = list(cw)
                    lst[pos] = new_val
                    indices.append(word_to_idx(lst))
        covered_map[cw_idx] = np.array(indices, dtype=np.int32)

    # Build coverage count using numpy for speed
    def build_coverage_count(cw_idx_list):
        counts = np.zeros(total_words, dtype=np.int32)
        for cw_idx in cw_idx_list:
            covered_indices = covered_map[cw_idx]
            np.add.at(counts, covered_indices, 1)
        return counts

    # Check if removing a codeword keeps coverage valid
    def is_redundant(cw_idx, counts):
        covered_indices = covered_map[cw_idx]
        return np.all(counts[covered_indices] > 1)

    # Initial construction: direct sum of ternary tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n >= 4:
        initial_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    else:
        # Fallback: start with all words
        initial_codewords = [list(word) for word in itertools.product(range(q), repeat=n)]

    # Convert to indices for faster processing
    def cw_list_to_idx_list(cw_list):
        return [word_to_idx(cw) for cw in cw_list]

    def idx_list_to_cw_list(idx_list):
        return [idx_to_word(idx) for idx in idx_list]

    initial_cw_idx_list = cw_list_to_idx_list(initial_codewords)

    best_cw_idx_list = None
    best_len = len(initial_cw_idx_list)

    # Multiple restarts with different strategies
    for restart in range(30):
        if time.time() > deadline:
            break

        cw_idx_list = list(initial_cw_idx_list)

        # Phase 1: Greedy removal (multiple passes)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            counts = build_coverage_count(cw_idx_list)

            # Try removing in reverse order
            for i in range(len(cw_idx_list) - 1, -1, -1):
                if time.time() > deadline:
                    break
                if is_redundant(cw_idx_list[i], counts):
                    cw_idx_list.pop(i)
                    improved = True
                    break

            if not improved and time.time() < deadline:
                # Try forward order
                counts = build_coverage_count(cw_idx_list)
                for i in range(len(cw_idx_list)):
                    if time.time() > deadline:
                        break
                    if is_redundant(cw_idx_list[i], counts):
                        cw_idx_list.pop(i)
                        improved = True
                        break

            if not improved and time.time() < deadline:
                # Try random order
                counts = build_coverage_count(cw_idx_list)
                indices = list(range(len(cw_idx_list)))
                random.shuffle(indices)
                for i in indices:
                    if time.time() > deadline:
                        break
                    if is_redundant(cw_idx_list[i], counts):
                        cw_idx_list.pop(i)
                        improved = True
                        break

        # Phase 2: Swap-based local search (add-then-remove)
        if time.time() < deadline:
            swap_improved = True
            max_swap_iters = 80
            for _ in range(max_swap_iters):
                if time.time() > deadline:
                    break
                if not swap_improved:
                    break
                swap_improved = False

                counts = build_coverage_count(cw_idx_list)
                # Find codewords that are "least useful" (cover fewest uniquely-covered words)
                unique_counts = []
                for i, cw_idx in enumerate(cw_idx_list):
                    covered = covered_map[cw_idx]
                    unique = np.sum(counts[covered] == 1)
                    unique_counts.append((unique, i))

                # Sort by unique coverage (least first)
                unique_counts.sort()

                # Try swapping the least useful codeword with a random new one
                for unique, i in unique_counts[:10]:
                    if time.time() > deadline:
                        break
                    if unique > 0:
                        continue  # Can't remove this one directly

                    # Try adding a random codeword and see if we can then remove more
                    new_cw_idx = random.randint(0, total_words - 1)
                    if new_cw_idx in cw_idx_list:
                        continue

                    # Temporarily add and remove
                    test_cw_idx_list = cw_idx_list.copy()
                    test_cw_idx_list.pop(i)
                    test_cw_idx_list.append(new_cw_idx)

                    # Check if this is still valid and allows further removals
                    test_counts = build_coverage_count(test_cw_idx_list)
                    if np.any(test_counts == 0):
                        continue  # Not valid

                    # Try to remove more from the test set
                    test_improved = True
                    while test_improved and time.time() < deadline:
                        test_improved = False
                        test_counts = build_coverage_count(test_cw_idx_list)
                        for j in range(len(test_cw_idx_list) - 1, -1, -1):
                            if time.time() > deadline:
                                break
                            if is_redundant(test_cw_idx_list[j], test_counts):
                                test_cw_idx_list.pop(j)
                                test_improved = True
                                break

                    if len(test_cw_idx_list) < len(cw_idx_list):
                        cw_idx_list = test_cw_idx_list
                        swap_improved = True
                        break

        # Update best
        if len(cw_idx_list) < best_len:
            best_len = len(cw_idx_list)
            best_cw_idx_list = cw_idx_list.copy()

        # If we're already at a good size, try more aggressive search
        if best_len <= 75 and time.time() < deadline:
            # Try a few more restarts with perturbation
            for pert in range(8):
                if time.time() > deadline:
                    break
                perturbed = best_cw_idx_list.copy()
                # Remove a random codeword and try to re-add a better one
                if len(perturbed) > 10:
                    remove_idx = random.randint(0, len(perturbed) - 1)
                    perturbed.pop(remove_idx)

                    # Try to fix coverage by adding random codewords
                    for _ in range(30):
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
                        # Pick a random codeword index that covers the uncovered word
                        # Find all codewords that cover unc_idx
                        candidates = []
                        for cw_idx in range(total_words):
                            if unc_idx in covered_map[cw_idx]:
                                candidates.append(cw_idx)
                        if candidates:
                            new_cw_idx = random.choice(candidates)
                            if new_cw_idx not in perturbed:
                                perturbed.append(new_cw_idx)

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
                                break

                    # Check validity
                    final_counts = build_coverage_count(perturbed)
                    if np.all(final_counts > 0) and len(perturbed) < best_len:
                        best_len = len(perturbed)
                        best_cw_idx_list = perturbed.copy()

    if best_cw_idx_list is None:
        best_cw_idx_list = initial_cw_idx_list

    return idx_list_to_cw_list(best_cw_idx_list)
# EVOLVE-BLOCK-END
