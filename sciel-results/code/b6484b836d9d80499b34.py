"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

import numpy as np


def _tetracode():
    # 9 words of length 4 over {0,1,2} forming a perfect code of covering radius 1.
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _direct_sum(q, n, radius):
    # Direct sum of the ternary tetracode with every word on the remaining n-4 coords.
    tetracode = _tetracode()
    if n == 4:
        return tetracode
    tails = itertools.product(range(3), repeat=n - 4)
    return [h + list(tail) for h in tetracode for tail in tails]


def _greedy_cover(q, n, radius):
    # Greedy set-cover: repeatedly pick the codeword that covers the most uncovered words.
    # For radius 1 over q=3, a codeword covers 1 + n*(q-1) = 1 + 2n words.
    # We use a priority-queue-like approach with numpy for speed.
    total = q ** n
    if total > 2_000_000:
        # Too large for full-space greedy; fall back to direct sum.
        return _direct_sum(q, n, radius)

    # Represent words as integers 0..q^n - 1
    # covered: boolean array of size total
    covered = np.zeros(total, dtype=np.bool_)
    codewords = []

    # Precompute: for each integer index, its digit representation (list of n ints)
    # And for each codeword index, the set of indices it covers (within radius)
    # For radius 1: a word covers itself and all words differing in exactly one position.

    # Map index -> digits
    def idx_to_digits(idx):
        d = []
        for _ in range(n):
            d.append(idx % q)
            idx //= q
        return d

    # Map digits -> index
    def digits_to_idx(d):
        idx = 0
        for i in range(n - 1, -1, -1):
            idx = idx * q + d[i]
        return idx

    # For a given word (as index), compute all indices within Hamming distance radius.
    # For radius=1: the word itself + n*(q-1) neighbors.
    def neighbors(idx):
        d = idx_to_digits(idx)
        result = [idx]
        for pos in range(n):
            orig = d[pos]
            for val in range(q):
                if val == orig:
                    continue
                d[pos] = val
                result.append(digits_to_idx(d))
            d[pos] = orig
        return result

    # Greedy: at each step, find the uncovered word that maximizes the number of
    # uncovered words within its radius-1 ball.
    # For efficiency, we maintain for each candidate the count of uncovered neighbors.
    # Initially every word has the same count (1 + n*(q-1)), so we can start with any.
    # After picking a codeword, we decrement counts for all words in its ball.

    # To avoid O(total^2) we use a simpler approach: iterate, and for each uncovered
    # word, count how many of its neighbors are still uncovered. Pick the max.
    # This is O(total * n * q) per step which is too slow for large total.

    # Alternative: since all balls have the same size, and we want to maximize overlap
    # reduction, a good heuristic is: pick the word that has the most uncovered neighbors.
    # We'll use a batch approach: maintain an array 'uncovered_count' for each word.

    # Actually for radius=1, q=3, n=8: total = 6561. Ball size = 1+16=17.
    # Greedy with 6561 candidates and ~486 steps: 6561*17*486 ≈ 54M operations. Feasible.

    # Precompute neighbor lists for all words
    print(f"  Precomputing neighbors for {total} words...")
    all_neighbors = []
    for idx in range(total):
        all_neighbors.append(neighbors(idx))

    # uncovered_count[i] = number of uncovered words in the ball of word i
    uncovered_count = np.array([len(nb) for nb in all_neighbors], dtype=np.int32)
    # Initially all are uncovered, so count = ball size = 1 + n*(q-1)
    ball_size = 1 + n * (q - 1)
    uncovered_count[:] = ball_size

    # We need to track which words are covered
    # covered is already a boolean array

    # Greedy loop
    max_iter = total  # safety
    for step in range(max_iter):
        if covered.all():
            break
        # Find the word with max uncovered_count that is not yet a codeword
        # We don't strictly need the word itself to be uncovered; we just need it to
        # cover the most uncovered words. But typically the best is uncovered.
        best_idx = int(np.argmax(uncovered_count))
        if uncovered_count[best_idx] <= 0:
            break

        # Add best_idx as a codeword
        codewords.append(idx_to_digits(best_idx))
        # Mark all words in its ball as covered, and decrement uncovered_count
        # for all words whose balls contain any of these newly covered words.
        # This is the expensive part. For each newly covered word w, we need to
        # decrement uncovered_count[v] for every v such that w is in ball(v).
        # Equivalently, for each v in all_neighbors, if v's ball intersects the newly
        # covered set, decrement.
        # Simpler: for each word v, uncovered_count[v] = number of uncovered words in ball(v).
        # When we cover a set S, for each v, decrement by |S ∩ ball(v)|.
        # S = all_neighbors[best_idx] (the ball of the chosen codeword)
        # So for each v, decrement by |all_neighbors[best_idx] ∩ all_neighbors[v]|.

        # This is O(total * ball_size) per step which is 6561*17 ≈ 111K per step,
        # times 486 steps ≈ 54M. OK.

        s = all_neighbors[best_idx]
        s_set = set(s)
        for v in range(total):
            # Count intersection of s_set and all_neighbors[v]
            inter = 0
            for w in all_neighbors[v]:
                if w in s_set:
                    inter += 1
            uncovered_count[v] -= inter

        # Mark covered
        for w in s:
            covered[w] = True

    return codewords


def construct(q, n, radius):
    deadline = time.time() + 110  # safety margin before 120s limit

    # For the specific case q=3, n=8, radius=1, try greedy.
    # For other cases, fall back to direct sum.
    if q == 3 and n == 8 and radius == 1:
        try:
            result = _greedy_cover(q, n, radius)
            # Verify validity
            total = q ** n
            if total <= 2_000_000:
                # Quick verification
                covered = np.zeros(total, dtype=np.bool_)
                for cw in result:
                    idx = 0
                    for i in range(n - 1, -1, -1):
                        idx = idx * q + cw[i]
                    # Mark ball
                    d = list(cw)
                    covered[idx] = True
                    for pos in range(n):
                        orig = d[pos]
                        for val in range(q):
                            if val == orig:
                                continue
                            d[pos] = val
                            ii = 0
                            for i in range(n - 1, -1, -1):
                                ii = ii * q + d[i]
                            covered[ii] = True
                        d[pos] = orig
                if not covered.all():
                    # Invalid, fall back
                    return _direct_sum(q, n, radius)
            return result
        except Exception:
            return _direct_sum(q, n, radius)
    else:
        return _direct_sum(q, n, radius)
# EVOLVE-BLOCK-END
