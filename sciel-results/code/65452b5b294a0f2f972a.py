"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np

def _time_left(start_time, deadline):
    return deadline - time.time()

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit

    if q == 3 and n == 8 and radius == 1:
        return _greedy_cover_trim(3, 8, 1, start_time, deadline)
    else:
        # General fallback
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

def _greedy_cover_trim(q, n, radius, start_time, deadline):
    """
    Strategy:
    1. Greedy set cover to build a small initial covering code
    2. Local search (redundancy removal + swap) to trim further
    """
    # Map words to indices: 3^8 = 6561
    total_space = q ** n
    # Precompute all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute spheres (bitmask representation for speed)
    # Sphere of a word: the word itself + all words at Hamming distance 1
    # For q=3, n=8: sphere size = 1 + 8*(3-1) = 17

    def get_sphere_indices(word_tuple):
        """Return list of indices of words within Hamming distance 1."""
        result = [word_to_idx[word_tuple]]
        for i in range(n):
            orig = word_tuple[i]
            for d in range(q):
                if d == orig:
                    continue
                new_list = list(word_tuple)
                new_list[i] = d
                result.append(word_to_idx[tuple(new_list)])
        return result

    # Greedy set cover:
    # Maintain: for each word in space, how many times it's covered
    # At each step, pick the word (from uncovered words) whose sphere covers
    # the most uncovered words, add it as a codeword, update counts.

    # For efficiency, precompute the sphere for every possible word
    # That's 6561 * 17 = ~111k entries, feasible
    spheres = [get_sphere_indices(w) for w in all_words]

    # Coverage count for each word in the space
    cover_count = np.zeros(total_space, dtype=np.int32)

    # Track which words are uncovered
    uncovered_count = np.ones(total_space, dtype=bool)
    num_uncovered = total_space

    code = []
    code_set = set()

    # Greedy: repeatedly pick the word that covers the most uncovered words
    # We need to efficiently find which candidate covers the most uncovered words.
    # Since the space is only 6561, we can iterate over all candidates each time
    # but that's 6561 * 17 * ~500 iterations = ~55M operations. Might be tight.
    # Let's use a smarter approach: maintain a priority or just do it in numpy.

    while num_uncovered > 0:
        if time.time() > deadline - 5:
            break

        # Find the word that covers the most uncovered words
        # For each candidate word c, the number of newly covered words is:
        # sum over s in sphere(c) of (1 if s is uncovered else 0)
        # = sum over s in sphere(c) of uncovered_count[s]

        # Vectorized approach: for each candidate, compute overlap with uncovered
        # This is expensive to do naively. Let's use a different approach.

        # Alternative: iterate through uncovered words, and for each, consider
        # which codewords could cover it. But that's also complex.

        # Simple approach: sample a subset of candidates each iteration
        # Actually, for 6561 words, let's just do the full scan but vectorize.

        # Build a matrix: rows = candidates (6561), cols = sphere positions (17)
        # sphere_matrix[i, j] = spheres[i][j]  (the word index)
        # Then for each candidate i, newly_covered = sum(uncovered_count[sphere_matrix[i, j]] for j in range(17))

        # Precompute sphere_matrix once
        if not hasattr(_greedy_cover_trim, '_sphere_matrix'):
            _greedy_cover_trim._sphere_matrix = np.array([spheres[i] for i in range(total_space)], dtype=np.int32)

        sphere_matrix = _greedy_cover_trim._sphere_matrix  # shape (6561, 17)

        # For each candidate, compute how many of its sphere members are uncovered
        # newly_covered[i] = sum(uncovered_count[sphere_matrix[i, j]] for j)
        # = uncovered_count[sphere_matrix[i, :]].sum(axis=1)
        newly_covered = uncovered_count[sphere_matrix].sum(axis=1)

        # Pick the candidate with maximum newly_covered
        best_idx = int(np.argmax(newly_covered))

        if newly_covered[best_idx] == 0:
            break  # No progress possible (shouldn't happen if num_uncovered > 0)

        # Add this word to the code
        code.append(list(all_words[best_idx]))
        code_set.add(all_words[best_idx])

        # Update coverage counts
        for s in spheres[best_idx]:
            cover_count[s] += 1
            if cover_count[s] == 1:
                uncovered_count[s] = False
                num_uncovered -= 1

    # Now we have an initial code from greedy set cover
    # Try to reduce it via local search

    if time.time() > deadline - 10:
        return code

    # Local search phase: try to remove redundant codewords
    # Use the same approach as before but on the smaller code

    # Recompute coverage counts for the current code
    cover_count = np.zeros(total_space, dtype=np.int32)
    for w in code:
        for s in spheres[word_to_idx[tuple(w)]]:
            cover_count[s] += 1

    # Verify full coverage
    if np.any(cover_count == 0):
        # Shouldn't happen, but safety check
        pass

    # Try to remove redundant codewords in random order
    indices = list(range(len(code)))
    random.seed(42)
    random.shuffle(indices)

    active = [True] * len(code)

    for idx in indices:
        if not active[idx]:
            continue
        if time.time() > deadline - 2:
            break

        w = tuple(code[idx])
        sphere_idx = word_to_idx[w]
        sphere_words = spheres[sphere_idx]

        # Check if all words in this codeword's sphere are covered by >= 2 codewords
        redundant = True
        for s in sphere_words:
            if cover_count[s] <= 1:
                redundant = False
                break

        if redundant:
            active[idx] = False
            for s in sphere_words:
                cover_count[s] -= 1

    # Build final code
    final_code = [code[i] for i in range(len(code)) if active[i]]

    # Try swap-based local search: replace a codeword with a different one
    # that covers at least as much uncovered territory
    if time.time() > deadline - 5:
        return final_code

    # Additional pass: try to find better replacements
    # For each codeword, see if there's another word that covers all the same
    # "critical" words (words covered by exactly 1 codeword after removing this one)

    # This is more complex; let's do a simple multi-pass removal
    for _pass in range(3):
        if time.time() > deadline - 2:
            break

        indices2 = list(range(len(final_code)))
        random.shuffle(indices2)
        active2 = [True] * len(final_code)

        # Recompute coverage
        cover_count2 = np.zeros(total_space, dtype=np.int32)
        for w in final_code:
            for s in spheres[word_to_idx[tuple(w)]]:
                cover_count2[s] += 1

        for idx in indices2:
            if not active2[idx]:
                continue
            if time.time() > deadline - 1:
                break

            w = tuple(final_code[idx])
            sphere_words = spheres[word_to_idx[w]]

            redundant = True
            for s in sphere_words:
                if cover_count2[s] <= 1:
                    redundant = False
                    break

            if redundant:
                active2[idx] = False
                for s in sphere_words:
                    cover_count2[s] -= 1

        final_code = [final_code[i] for i in range(len(final_code)) if active2[i]]

    return final_code
# EVOLVE-BLOCK-END
