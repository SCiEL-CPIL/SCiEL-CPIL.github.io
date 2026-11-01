import itertools
import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    # Precompute utilities for q=3, n=8
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters: simple greedy
        return _general_greedy(q, n, radius, deadline - start_time)

    total_words = 3 ** n
    # Represent words as integers 0 to 6560
    # Precompute balls: for each word, list of words in its ball of radius 1
    # Ball size is 1 + 8 * 2 = 17

    balls = np.empty((total_words, 17), dtype=np.int32)

    # Convert int to base-3 digits helper
    def int_to_base3(x, n):
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = x % 3
            x //= 3
        return digits

    def base3_to_int(digits):
        res = 0
        for d in digits:
            res = res * 3 + d
        return res

    # Precomputation
    for x in range(total_words):
        digits = int_to_base3(x, n)
        ball_indices = [x]
        for j in range(n):
            for v in range(3):
                if v != digits[j]:
                    new_digits = digits[:]
                    new_digits[j] = v
                    ball_indices.append(base3_to_int(new_digits))
        balls[x] = ball_indices

    best_codewords = None
    best_size = total_words + 1

    # Multiple randomized restarts
    num_restarts = 5
    time_per_restart = (deadline - start_time - 10) / num_restarts

    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break

        random.seed(42 + restart)
        remaining = np.ones(total_words, dtype=bool)
        codewords = []

        # Greedy cover with randomization
        while remaining.any():
            if time.time() > deadline - 15:
                break

            # Vectorized count of uncovered words in each ball
            counts = remaining[balls].sum(axis=1)
            max_count = counts.max()

            if max_count == 0:
                break

            # Get all candidates with max count
            max_indices = np.where(counts == max_count)[0]

            if len(max_indices) == 1:
                best_candidate = max_indices[0]
            else:
                # Random tie-breaking with slight preference for uncovered candidates
                # Weight by whether the candidate itself is uncovered
                weights = remaining[max_indices].astype(float)
                # Add small random noise to break ties
                noise = random.random() * 0.1
                weights += noise
                best_candidate = max_indices[np.argmax(weights)]

            codewords.append(best_candidate)
            remaining[balls[best_candidate]] = False

        # If we didn't finish due to time, complete the cover
        if remaining.any():
            while remaining.any():
                if time.time() > deadline - 5:
                    break
                idx = np.where(remaining)[0][0]
                codewords.append(idx)
                remaining[balls[idx]] = False

        # Redundancy removal - more aggressive
        if time.time() < deadline - 10:
            codewords = _remove_redundancy(codewords, balls, total_words, deadline)

        # Track best
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_codewords = codewords.copy()

        # If we've reached the known optimal, stop early
        if best_size <= 486:
            break

    if best_codewords is None:
        # Fallback: return all words (valid but large)
        return [int_to_base3(i, n) for i in range(total_words)]

    return [int_to_base3(c, n) for c in best_codewords]

def _remove_redundancy(codewords, balls, total_words, deadline):
    """Remove redundant codewords using iterative approach."""
    if not codewords:
        return codewords

    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in codewords:
        cover_count[balls[c]] += 1

    changed = True
    max_iterations = 20
    iteration = 0

    while changed and iteration < max_iterations:
        if time.time() > deadline - 5:
            break
        changed = False
        new_codewords = []
        for c in codewords:
            # Check if c is redundant
            if cover_count[balls[c]].min() >= 2:
                # Redundant
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_codewords.append(c)
        codewords = new_codewords
        iteration += 1

    return codewords

def _general_greedy(q, n, radius, time_limit):
    # Simple fallback for non-3,8,1 cases
    # Not optimized, but functional
    total = q ** n
    covered = np.zeros(total, dtype=bool)
    codewords = []

    def int_to_word(x):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = x % q
            x //= q
        return w

    def word_to_int(w):
        res = 0
        for d in w:
            res = res * q + d
        return res

    # Generate balls on the fly or precompute if small
    # For general case, precomputation might be heavy if q^n is large.
    # But task is specific to 3,8,1.
    # This is just a safety net.

    start = time.time()
    while covered.sum() < total:
        if time.time() - start > time_limit - 5:
            break
        best_c = -1
        best_count = -1
        for c in range(total):
            if covered[c]: continue # Optimization: if c is covered, it might still cover others, but usually we pick uncovered ones? No, pick max coverage.
            # Calculate coverage
            w = int_to_word(c)
            count = 0
            # Check self
            if not covered[c]: count += 1
            # Check neighbors
            for j in range(n):
                for v in range(q):
                    if v == w[j]: continue
                    w2 = w[:]
                    w2[j] = v
                    idx = word_to_int(w2)
                    if not covered[idx]: count += 1
            if count > best_count:
                best_count = count
                best_c = c

        if best_c == -1: break
        codewords.append(best_c)
        # Update covered
        w = int_to_word(best_c)
        covered[best_c] = True
        for j in range(n):
            for v in range(q):
                if v == w[j]: continue
                w2 = w[:]
                w2[j] = v
                covered[word_to_int(w2)] = True

    return [int_to_word(c) for c in codewords]
