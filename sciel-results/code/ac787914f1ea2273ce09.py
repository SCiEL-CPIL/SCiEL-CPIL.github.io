"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import numpy as np
import time


def _encode_words(words, q, n):
    """Encode list of words (ints) into base-q integer indices."""
    if n == 0:
        return [0] * len(words)
    powers = np.array([q ** i for i in range(n)], dtype=np.int64)
    arr = np.array(words, dtype=np.int64)
    return (arr @ powers).tolist()


def _decode_idx(idx, q, n):
    """Decode base-q index into word tuple."""
    if n == 0:
        return ()
    powers = [q ** i for i in range(n)]
    word = [0] * n
    for i in range(n):
        word[i] = idx % q
        idx //= q
    return tuple(word)


def construct(q, n, radius):
    if n == 0:
        return [[]]

    # For small cases, use direct construction
    if n <= 4:
        if n == 4:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        elif n == 3:
            return [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        elif n == 2:
            return [[0, 0], [1, 1], [2, 2], [0, 1], [1, 0], [2, 2], [0, 2], [2, 0], [1, 2], [2, 1]]
        else:
            return [[i] for i in range(q)]

    # Main case: use vectorized greedy + local search
    # Precompute all word indices
    all_indices = np.arange(q ** n, dtype=np.int64)
    all_words_arr = np.zeros((len(all_indices), n), dtype=np.int8)
    for i in range(n):
        all_words_arr[:, i] = (all_indices // (q ** i)) % q

    # Precompute neighborhood for each word index (vectorized per position)
    # For radius=1, neighbors are: self + change one coordinate to q-1 other values
    # We'll compute coverage on the fly using vectorized operations

    # Start with tetracode-based initial solution
    tetracode = np.array([[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)], dtype=np.int8)
    if n == 4:
        initial_arr = tetracode
    else:
        tails = np.array(list(itertools.product(range(3), repeat=n - 4)), dtype=np.int8)
        initial_arr = np.concatenate([np.repeat(tetracode, len(tails), axis=0),
                                      np.tile(tails, (len(tetracode), 1))], axis=1)

    codewords = [list(w) for w in initial_arr]
    cw_indices = _encode_words(codewords, q, n)

    # Coverage count: how many codewords cover each word
    covered_count = np.zeros(len(all_indices), dtype=np.int32)

    def update_coverage(idx_list, sign):
        """Update covered_count for a list of codeword indices."""
        for cw_idx in idx_list:
            cw = all_words_arr[cw_idx]
            # Self
            covered_count[cw_idx] += sign
            # Change each coordinate
            for pos in range(n):
                for val in range(q):
                    if val != cw[pos]:
                        new_word = cw.copy()
                        new_word[pos] = val
                        new_idx = int(np.dot(new_word, [q ** i for i in range(n)]))
                        covered_count[new_idx] += sign

    # Initialize coverage
    for cw_idx in cw_indices:
        cw = all_words_arr[cw_idx]
        covered_count[cw_idx] += 1
        for pos in range(n):
            for val in range(q):
                if val != cw[pos]:
                    new_word = cw.copy()
                    new_word[pos] = val
                    new_idx = int(np.dot(new_word, [q ** i for i in range(n)]))
                    covered_count[new_idx] += 1

    # Greedy: add codewords that cover most uncovered words
    uncovered = np.where(covered_count == 0)[0]
    max_greedy_steps = 200
    for step in range(max_greedy_steps):
        if len(uncovered) == 0:
            break
        if len(uncovered) < 10:
            # Try to cover remaining with targeted search
            break
        # Find best codeword to add
        best_cw = -1
        best_gain = -1
        # Sample candidates if space is large
        if len(all_indices) <= 5000:
            candidates = all_indices
        else:
            candidates = np.random.choice(all_indices, size=min(500, len(all_indices)), replace=False)
        for cw_idx in candidates:
            if covered_count[cw_idx] > 0:
                # Check if this codeword is already selected
                if cw_idx in cw_indices:
                    continue
            cw = all_words_arr[cw_idx]
            # Count how many uncovered words this covers
            gain = 0
            if covered_count[cw_idx] == 0:
                gain += 1
            for pos in range(n):
                for val in range(q):
                    if val != cw[pos]:
                        new_word = cw.copy()
                        new_word[pos] = val
                        new_idx = int(np.dot(new_word, [q ** i for i in range(n)]))
                        if covered_count[new_idx] == 0:
                            gain += 1
            if gain > best_gain:
                best_gain = gain
                best_cw = cw_idx
        if best_cw == -1 or best_gain <= 0:
            break
        codewords.append([int(x) for x in all_words_arr[best_cw]])
        cw_indices.append(best_cw)
        # Update coverage
        cw = all_words_arr[best_cw]
        covered_count[best_cw] += 1
        for pos in range(n):
            for val in range(q):
                if val != cw[pos]:
                    new_word = cw.copy()
                    new_word[pos] = val
                    new_idx = int(np.dot(new_word, [q ** i for i in range(n)]))
                    covered_count[new_idx] += 1
        uncovered = np.where(covered_count == 0)[0]

    # Local search: try to remove codewords or replace
    best_codewords = codewords[:]
    best_size = len(codewords)
    random.seed(42)
    start_time = time.time()
    time_limit = 110  # seconds

    # First pass: try removing redundant codewords
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(len(best_codewords)):
            cw_idx = cw_indices[i]
            # Check if removing this codeword leaves all words covered
            cw = all_words_arr[cw_idx]
            can_remove = True
            # Check self
            if covered_count[cw_idx] == 1:
                can_remove = False
            else:
                # Check neighbors
                for pos in range(n):
                    for val in range(q):
                        if val != cw[pos]:
                            new_word = cw.copy()
                            new_word[pos] = val
                            new_idx = int(np.dot(new_word, [q ** i for i in range(n)]))
                            if covered_count[new_idx] == 1:
                                can_remove = False
                                break
                    if not can_remove:
                        break
            if can_remove:
                # Remove codeword
                del best_codewords[i]
                del cw_indices[i]
                # Update coverage
                cw = all_words_arr[cw_idx]
                covered_count[cw_idx] -= 1
                for pos in range(n):
                    for val in range(q):
                        if val != cw[pos]:
                            new_word = cw.copy()
                            new_word[pos] = val
                            new_idx = int(np.dot(new_word, [q ** i for i in range(n)]))
                            covered_count[new_idx] -= 1
                best_size = len(best_codewords)
                improved = True
                break

    # Second pass: try 1-for-1 replacements
    for iteration in range(100):
        if time.time() - start_time > time_limit:
            break
        if len(best_codewords) <= best_size:
            break
        # Try to find a replacement that allows removal of another codeword
        for i in range(len(best_codewords)):
            if time.time() - start_time > time_limit:
                break
            cw_idx = cw_indices[i]
            cw = all_words_arr[cw_idx]
            # Find words that would become uncovered if we remove this
            would_uncover = []
            if covered_count[cw_idx] == 1:
                would_uncover.append(cw_idx)
            for pos in range(n):
                for val in range(q):
                    if val != cw[pos]:
                        new_word = cw.copy()
                        new_word[pos] = val
                        new_idx = int(np.dot(new_word, [q ** i for i in range(n)]))
                        if covered_count[new_idx] == 1:
                            would_uncover.append(new_idx)
            if not would_uncover:
                # Can remove
                del best_codewords[i]
                del cw_indices[i]
                cw = all_words_arr[cw_idx]
                covered_count[cw_idx] -= 1
                for pos in range(n):
                    for val in range(q):
                        if val != cw[pos]:
                            new_word = cw.copy()
                            new_word[pos] = val
                            new_idx = int(np.dot(new_word, [q ** i for i in range(n)]))
                            covered_count[new_idx] -= 1
                best_size = len(best_codewords)
                break
            else:
                # Try to find a candidate that covers all would_uncover words
                # and is not already in the code
                # Sample candidates
                if len(all_indices) <= 5000:
                    candidates = all_indices
                else:
                    candidates = np.random.choice(all_indices, size=min(200, len(all_indices)), replace=False)
                found = False
                for cand_idx in candidates:
                    if cand_idx in cw_indices:
                        continue
                    cand = all_words_arr[cand_idx]
                    covers_all = True
                    for w_idx in would_uncover:
                        w = all_words_arr[w_idx]
                        dist = np.sum(cand != w)
                        if dist > radius:
                            covers_all = False
                            break
                    if covers_all:
                        # Replace codeword i with candidate
                        old_idx = cw_indices[i]
                        # Remove old
                        cw_old = all_words_arr[old_idx]
                        covered_count[old_idx] -= 1
                        for pos in range(n):
                            for val in range(q):
                                if val != cw_old[pos]:
                                    new_word = cw_old.copy()
                                    new_word[pos] = val
                                    new_idx = int(np.dot(new_word, [q ** i for i in range(n)]))
                                    covered_count[new_idx] -= 1
                        # Add new
                        covered_count[cand_idx] += 1
                        for pos in range(n):
                            for val in range(q):
                                if val != cand[pos]:
                                    new_word = cand.copy()
                                    new_word[pos] = val
                                    new_idx = int(np.dot(new_word, [q ** i for i in range(n)]))
                                    covered_count[new_idx] += 1
                        best_codewords[i] = [int(x) for x in cand]
                        cw_indices[i] = cand_idx
                        found = True
                        break
                if found:
                    break

    return best_codewords
# EVOLVE-BLOCK-END
