"""Combined program: vectorized greedy + enhanced local search for ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    # Total number of words in the space
    total_words = q ** n

    # Build index mapping: integer -> tuple of digits, and tuple -> integer
    word_to_idx = {}
    idx_to_word = {}
    for i in range(total_words):
        digits = []
        x = i
        for _ in range(n):
            digits.append(x % q)
            x //= q
        word_to_idx[tuple(digits)] = i
        idx_to_word[i] = tuple(digits)

    # Precompute neighbors (ball of radius 1) for each word as list of indices
    # neighbors[i] = list of j such that Hamming distance(idx_to_word[i], idx_to_word[j]) <= radius
    neighbors = []
    neighbor_sets = []
    for i in range(total_words):
        w = list(idx_to_word[i])
        neigh = [i]
        if radius == 1:
            for pos in range(n):
                orig_val = w[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    w[pos] = val
                    idx = word_to_idx[tuple(w)]
                    neigh.append(idx)
                w[pos] = orig_val
        else:
            # For general radius, generate all words within distance radius
            def _gen(idx, current, changes_left):
                if changes_left == 0:
                    neigh.append(word_to_idx[tuple(current)])
                    return
                # Don't change this position
                _gen(idx + 1, current, changes_left)
                # Change this position
                orig = current[idx]
                for val in range(q):
                    if val != orig:
                        current[idx] = val
                        _gen(idx + 1, current, changes_left - 1)
                current[idx] = orig
            _gen(0, list(idx_to_word[i]), radius)
            # Remove duplicates
            neigh = list(set(neigh))
        neighbors.append(neigh)
        neighbor_sets.append(set(neigh))

    # Convert neighbors to numpy arrays for vectorized operations
    neighbor_arrays = [np.array(neigh, dtype=np.int32) for neigh in neighbors]

    # Phase 1: Greedy Set Cover with vectorized gain computation
    uncovered = np.ones(total_words, dtype=bool)
    code = []
    code_set = set()

    while np.any(uncovered):
        if time.time() > deadline - 30:
            break

        # Compute gains: for each candidate codeword, how many uncovered points does it cover?
        # Do this by accumulating from uncovered points
        gains = np.zeros(total_words, dtype=np.int32)
        uncovered_indices = np.where(uncovered)[0]
        for idx in uncovered_indices:
            # All codewords within distance radius of idx cover idx
            gains[neighbor_arrays[idx]] += 1

        best_idx = int(np.argmax(gains))
        max_gain = gains[best_idx]

        if max_gain == 0:
            break

        code.append(best_idx)
        code_set.add(best_idx)
        uncovered[neighbor_arrays[best_idx]] = False

    # Phase 2: Local Search - deletion pass and swap pass
    # Initialize coverage counts
    counts = np.zeros(total_words, dtype=np.int32)
    for idx in code:
        counts[neighbor_arrays[idx]] += 1

    best_size = len(code)
    best_code = list(code)

    # Iterative local search
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False

        # --- Deletion Pass ---
        to_remove = []
        for c in code:
            if np.min(counts[neighbor_arrays[c]]) > 1:
                to_remove.append(c)

        if to_remove:
            improved = True
            for c in to_remove:
                code.remove(c)
                code_set.remove(c)
                counts[neighbor_arrays[c]] -= 1

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        # --- Swap Pass ---
        if time.time() > deadline - 10:
            break

        random.shuffle(code)

        for c in code:
            if time.time() > deadline - 10:
                break

            # Check if c is currently removable
            if np.min(counts[neighbor_arrays[c]]) > 1:
                continue

            # Get critical points for c: points in neighbors[c] with counts == 1
            crit_points = [i for i in neighbors[c] if counts[i] == 1]

            if not crit_points:
                continue

            # Strategy 1: Find replacement covering all critical points
            found_swap = False
            random.shuffle(crit_points)

            for p in crit_points:
                if found_swap:
                    break

                cands = [x for x in neighbors[p] if x not in code_set]

                best_w = -1
                best_score = -1

                for w in cands:
                    valid = True
                    for k in crit_points:
                        if k not in neighbor_sets[w]:
                            valid = False
                            break

                    if not valid:
                        continue

                    # Score: maximize redundancy creation
                    score = int(np.sum(counts[neighbor_arrays[w]] == 2))
                    if score > best_score:
                        best_score = score
                        best_w = w

                if best_w != -1:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbor_arrays[c]] -= 1

                    code.append(best_w)
                    code_set.add(best_w)
                    counts[neighbor_arrays[best_w]] += 1

                    improved = True
                    found_swap = True
                    break

            # Strategy 2: Relocation - try to move c to cover would-be-uncovered points
            if not found_swap:
                temp_counts = counts.copy()
                temp_counts[neighbor_arrays[c]] -= 1

                if np.min(temp_counts) >= 1:
                    continue

                would_be_uncovered = [i for i in neighbors[c] if temp_counts[i] == 0]

                if not would_be_uncovered:
                    continue

                all_needed = set(crit_points) | set(would_be_uncovered)

                best_w = -1
                best_score = -1

                first_point = list(all_needed)[0]
                cands = [x for x in neighbors[first_point] if x not in code_set]

                for w in cands:
                    valid = True
                    for k in all_needed:
                        if k not in neighbor_sets[w]:
                            valid = False
                            break
                    if not valid:
                        continue

                    score = int(np.sum(counts[neighbor_arrays[w]] == 2))
                    if score > best_score:
                        best_score = score
                        best_w = w

                if best_w != -1:
                    code.remove(c)
                    code_set.remove(c)
                    counts[neighbor_arrays[c]] -= 1

                    code.append(best_w)
                    code_set.add(best_w)
                    counts[neighbor_arrays[best_w]] += 1

                    improved = True

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

    # Phase 3: Try a second greedy restart with different randomization if time permits
    if time.time() < deadline - 25 and best_size > 180:
        # Second greedy pass with perturbation
        uncovered2 = np.ones(total_words, dtype=bool)
        code2 = []
        code_set2 = set()

        # Add some random initial codewords to break symmetry
        num_init = min(10, total_words // 100)
        init_candidates = list(range(total_words))
        random.shuffle(init_candidates)
        for idx in init_candidates[:num_init]:
            code2.append(idx)
            code_set2.add(idx)
            uncovered2[neighbor_arrays[idx]] = False

        while np.any(uncovered2) and time.time() < deadline - 15:
            gains2 = np.zeros(total_words, dtype=np.int32)
            uncovered_indices2 = np.where(uncovered2)[0]
            for idx in uncovered_indices2:
                gains2[neighbor_arrays[idx]] += 1

            best_idx2 = int(np.argmax(gains2))
            max_gain2 = gains2[best_idx2]

            if max_gain2 == 0:
                break

            code2.append(best_idx2)
            code_set2.add(best_idx2)
            uncovered2[neighbor_arrays[best_idx2]] = False

        # Local search on second solution
        if time.time() < deadline - 10:
            counts2 = np.zeros(total_words, dtype=np.int32)
            for idx in code2:
                counts2[neighbor_arrays[idx]] += 1

            improved2 = True
            while improved2 and time.time() < deadline - 5:
                improved2 = False

                to_remove2 = []
                for c in code2:
                    if np.min(counts2[neighbor_arrays[c]]) > 1:
                        to_remove2.append(c)

                if to_remove2:
                    improved2 = True
                    for c in to_remove2:
                        code2.remove(c)
                        code_set2.remove(c)
                        counts2[neighbor_arrays[c]] -= 1

                if len(code2) < best_size:
                    best_size = len(code2)
                    best_code = list(code2)

    # Convert best_code back to list of lists
    result = [list(idx_to_word[cw]) for cw in best_code]

    return result
# EVOLVE-BLOCK-END
